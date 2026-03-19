# src/run_codereval_one_real.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError:
    print("[FATAL] Missing dependency: pyyaml")
    raise


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def p(title: str, value: Any = None) -> None:
    print("\n" + "=" * 100)
    print(title)
    if value is not None:
        if isinstance(value, (dict, list)):
            print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(value)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a dict: {path}")
    return data


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_pipeline():
    from beacon_system.pipeline import run_pipeline  # type: ignore
    return run_pipeline


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def extract_code_block(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def select_task(tasks: Any, task_id: Optional[str], index: int) -> Dict[str, Any]:
    # case 1: wrapped CoderEval format {"RECORDS": [...]}
    if isinstance(tasks, dict) and "RECORDS" in tasks and isinstance(tasks["RECORDS"], list):
        tasks = tasks["RECORDS"]

    # case 2: direct list[dict]
    if isinstance(tasks, list):
        dict_items = [x for x in tasks if isinstance(x, dict)]
        if not dict_items:
            raise ValueError("Task file is a list, but contains no dict tasks.")

        if task_id:
            for item in dict_items:
                if str(item.get("task_id") or item.get("id") or item.get("_id") or "") == task_id:
                    return item
            raise KeyError(f"task_id not found: {task_id}")

        if index < 0 or index >= len(dict_items):
            raise IndexError(f"Task index out of range: {index}")
        return dict_items[index]

    # case 3: direct dict task
    if isinstance(tasks, dict):
        if any(k in tasks for k in ("task_id", "id", "_id", "prompt", "docstring", "code")):
            return tasks

    raise ValueError(f"Unsupported task file structure: {type(tasks).__name__}")


def pick(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def normalize_path_like(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(value).replace("\\", "/").strip()


def infer_target_file(task: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return normalize_path_like(override)  # type: ignore

    candidates = [
        pick(task, "target_file", "file", "file_path", "path"),
        pick(task.get("target") or {}, "file", "path", "file_path"),
        pick(task.get("metadata") or {}, "target_file", "file", "path"),
    ]
    for c in candidates:
        if c:
            return normalize_path_like(str(c))  # type: ignore

    raise ValueError("Cannot infer target_file from task. Please pass --target-file.")


def infer_target_qualname(task: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return override.strip()

    candidates = [
        pick(task, "target_qualname", "qualname", "function_name", "method_name", "symbol"),
        pick(task.get("target") or {}, "qualname", "symbol", "function_name", "method_name"),
        pick(task.get("metadata") or {}, "target_qualname", "qualname", "symbol"),
    ]
    for c in candidates:
        if c:
            return str(c).strip()

    raise ValueError("Cannot infer target_qualname from task. Please pass --target-qualname.")


def infer_spec_text(task: Dict[str, Any]) -> str:
    pieces: List[str] = []

    for key in ("prompt", "instruction", "description", "docstring", "spec", "task_description"):
        v = task.get(key)
        if isinstance(v, str) and v.strip():
            pieces.append(v.strip())

    meta = task.get("metadata") or {}
    if isinstance(meta, dict):
        for key in ("prompt", "instruction", "description", "docstring", "spec"):
            v = meta.get(key)
            if isinstance(v, str) and v.strip():
                pieces.append(v.strip())

    if not pieces:
        return "Implement the target function/method so that the project tests pass."

    return "\n\n".join(dict.fromkeys(pieces))


def infer_signature_from_source(source: str, qualname: str) -> Optional[str]:
    try:
        tree = ast.parse(source)
    except Exception:
        return None

    parts = qualname.split(".")
    if len(parts) == 1:
        fn_name = parts[0]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
                seg = ast.get_source_segment(source, node)
                if seg:
                    first = seg.strip().splitlines()[0]
                    return first.rstrip(":") + ":"
        return None

    if len(parts) == 2:
        cls_name, fn_name = parts
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == fn_name:
                        seg = ast.get_source_segment(source, sub)
                        if seg:
                            first = seg.strip().splitlines()[0]
                            return first.rstrip(":") + ":"
        return None

    return None


def build_agent_task_from_codereval(
    raw_task: Dict[str, Any],
    source_text: str,
    target_file: str,
    target_qualname: str,
) -> Dict[str, Any]:
    signature = infer_signature_from_source(source_text, target_qualname) or f"def {target_qualname.split('.')[-1]}(...):"

    doc = (
        str(raw_task.get("docstring") or "").strip()
        or str(raw_task.get("prompt") or "").strip()
        or "Implement the target function/method so that the project tests pass."
    )

    extra_context = []
    if raw_task.get("all_context"):
        extra_context.append(f"Additional benchmark context:\n{raw_task['all_context']}")
    if raw_task.get("dependency"):
        extra_context.append(f"Dependency info:\n{raw_task['dependency']}")

    context_blocks = [
        "You are patching a real CoderEval task in an existing codebase.",
        "Return only the target function or method implementation.",
        "Do not rewrite the whole file.",
        "Keep imports unchanged unless strictly necessary.",
        "Use the existing project context below.",
        source_text,
        *extra_context,
    ]

    task_id = str(
        raw_task.get("task_id")
        or raw_task.get("id")
        or raw_task.get("_id")
        or "codereval_real_single"
    )

    return {
        "task_id": task_id,
        "lang": "python",
        "entry_function": target_qualname.split(".")[-1],
        "signature": signature,
        "docstring": doc,
        "context_blocks": context_blocks,
        "runnable_level": "project_runnable",
        "project_context": {
            "target_file": target_file,
            "constraints": [
                f"must_define:{target_qualname.split('.')[-1]}",
                "must_be_python",
                "deterministic_only",
                "no_network",
                "no_file_io",
            ],
            "tests": [],
        },
    }

def extract_target_definition(full_code: str, qualname: str) -> str:
    """
    Extract only the target def from generated content.
    Supports:
    - top-level function: func
    - class method: Class.method
    """
    code = extract_code_block(full_code)
    parts = qualname.split(".")
    target_name = parts[-1]

    try:
        tree = ast.parse(code)
        if len(parts) == 1:
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name:
                    return ast.unparse(node)
        elif len(parts) == 2:
            cls_name, fn_name = parts
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == cls_name:
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == fn_name:
                            return ast.unparse(sub)
            # allow model to return top-level def for a method target; caller can decide not to use it
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name:
                    return ast.unparse(node)
    except Exception:
        pass

    return code.strip()


def locate_def_span(source: str, qualname: str) -> Tuple[int, int, int]:
    """
    Return:
    - start line index (0-based, inclusive)
    - end line index (0-based, exclusive)
    - indent spaces to apply to replacement
    """
    tree = ast.parse(source)
    parts = qualname.split(".")

    if len(parts) == 1:
        fn_name = parts[0]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
                if node.end_lineno is None:
                    raise RuntimeError("AST node missing end_lineno")
                return node.lineno - 1, node.end_lineno, 0

    elif len(parts) == 2:
        cls_name, fn_name = parts
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == fn_name:
                        if sub.end_lineno is None:
                            raise RuntimeError("AST node missing end_lineno")
                        indent = sub.col_offset
                        return sub.lineno - 1, sub.end_lineno, indent

    raise RuntimeError(f"Cannot locate target definition span for {qualname}")


def indent_block(text: str, spaces: int) -> str:
    if spaces <= 0:
        return text.strip() + "\n"
    prefix = " " * spaces
    lines = text.strip().splitlines()
    return "\n".join(prefix + line if line.strip() else "" for line in lines) + "\n"


def patch_source_with_generated_def(source: str, generated_def: str, qualname: str) -> str:
    start, end, indent = locate_def_span(source, qualname)
    lines = source.splitlines(keepends=True)
    replacement = indent_block(generated_def, indent)
    return "".join(lines[:start]) + replacement + "".join(lines[end:])


def run_cmd(cwd: Path, cmd: str) -> Dict[str, Any]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        shell=True,
        capture_output=True,
        text=True,
    )
    return {
        "cmd": cmd,
        "cwd": str(cwd),
        "returncode": proc.returncode,
        "status": "pass" if proc.returncode == 0 else "fail",
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Beacon agent -> CoderEval single real task integration")
    parser.add_argument("--config", default=str(REPO_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--task-file", required=True, help="Path to CoderEval JSON task file")
    parser.add_argument("--task-id", default=None, help="Select task by task_id/id")
    parser.add_argument("--task-index", type=int, default=0, help="Select task by index if no task_id")
    parser.add_argument("--repo-root", required=True, help="Real project repo root for this task")
    parser.add_argument("--target-file", default=None, help="Override target file path relative to repo root")
    parser.add_argument("--target-qualname", default=None, help="Override target qualname, e.g. foo or Class.method")
    parser.add_argument("--run-cmd", default="pytest -q", help="Command to run project tests")
    args = parser.parse_args()

    try:
        p("[Step 0] Repo root", str(REPO_ROOT))

        config_path = Path(args.config)
        config = load_yaml(config_path)
        p("[Step 1] Loaded config path", str(config_path))
        p("[Step 1.1] Config content", config)

        run_pipeline = resolve_pipeline()
        p("[Step 2] Resolved pipeline entry", "beacon_system.pipeline.run_pipeline")

        raw_tasks = load_json(Path(args.task_file))
        raw_task = select_task(raw_tasks, args.task_id, args.task_index)
        p("[Step 3] Selected raw CoderEval task", raw_task)

        real_repo_root = Path(args.repo_root).resolve()
        if not real_repo_root.exists():
            raise FileNotFoundError(f"repo_root not found: {real_repo_root}")
        p("[Step 4] Real repo root", str(real_repo_root))

        target_file_rel = infer_target_file(raw_task, args.target_file)
        target_qualname = infer_target_qualname(raw_task, args.target_qualname)
        target_file_abs = real_repo_root / target_file_rel

        p("[Step 4.1] Resolved target file", str(target_file_abs))
        p("[Step 4.2] Resolved target qualname", target_qualname)

        if not target_file_abs.exists():
            raise FileNotFoundError(f"target_file not found in repo: {target_file_abs}")

        original_source = read_text(target_file_abs)
        p("[Step 5] Original target source file", original_source)

        agent_task = build_agent_task_from_codereval(
            raw_task=raw_task,
            source_text=original_source,
            target_file=target_file_rel,
            target_qualname=target_qualname,
        )
        p("[Step 6] Built Beacon agent task", agent_task)

        result = run_pipeline(task=agent_task, config=config)
        p("[Step 7] Pipeline returned", result)

        run_dir = Path(result["run_dir"])
        p("[Step 7.1] Run dir", str(run_dir))

        generated_raw = read_text(run_dir / "code_round1.py")
        verifier_json = read_text(run_dir / "verifier_round1.json")
        exec_json = read_text(run_dir / "exec_round1.json")

        p("[Step 8] code_round1.py", generated_raw)
        p("[Step 8.1] verifier_round1.json", verifier_json)
        p("[Step 8.2] exec_round1.json", exec_json)

        generated_def = extract_target_definition(generated_raw, target_qualname)
        p("[Step 9] Extracted generated target definition", generated_def)

        # temp workspace
        with tempfile.TemporaryDirectory(prefix="beacon_codereval_real_") as tmp:
            work_root = Path(tmp) / "repo_copy"
            shutil.copytree(real_repo_root, work_root)
            p("[Step 10] Temporary work repo", str(work_root))

            work_target_file = work_root / target_file_rel
            work_original_source = read_text(work_target_file)
            patched_source = patch_source_with_generated_def(
                source=work_original_source,
                generated_def=generated_def,
                qualname=target_qualname,
            )

            p("[Step 10.1] Patched target source", patched_source)

            write_text(work_target_file, patched_source)
            p("[Step 10.2] Wrote patched file", str(work_target_file))

            baseline = run_cmd(work_root, args.run_cmd)
            p("[Step 11] Test result on patched repo", baseline)

            final_summary = {
                "pipeline_status": result.get("status"),
                "pipeline_mode": result.get("mode"),
                "verifier_ok": result.get("verifier_ok"),
                "exec_status": result.get("exec_status"),
                "target_file": target_file_rel,
                "target_qualname": target_qualname,
                "patched_repo_test_status": baseline["status"],
                "patched_repo_test_returncode": baseline["returncode"],
                "run_dir": str(run_dir),
                "work_repo": str(work_root),
            }
            p("[Step 12] Final summary", final_summary)

            if result.get("status") != "ok":
                raise RuntimeError(f"Pipeline failed: {result}")

            if baseline["status"] != "pass":
                raise RuntimeError("Patched real repo tests did not pass.")

            print("\n" + "=" * 100)
            print("CODEREVAL_ONE_REAL_OK")
            print("=" * 100)
            return 0

    except Exception as e:
        print("\n" + "=" * 100)
        print("[FATAL] CoderEval one-real integration failed")
        print(repr(e))
        traceback.print_exc()
        print("=" * 100)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())