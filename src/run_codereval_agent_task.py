# src/run_codereval_agent_task.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


# -----------------------------------------------------------------------------
# Path bootstrap
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
DEFAULT_CODEREVAL_ROOT = REPO_ROOT / "benchmarks" / "CoderEval"
DEFAULT_JSON_PATH = DEFAULT_CODEREVAL_ROOT / "CoderEval4Python.json"
DEFAULT_WORKSPACE = DEFAULT_CODEREVAL_ROOT / "workspace"
DEFAULT_PROJECT_ROOT = DEFAULT_WORKSPACE / "neo4j-python-driver"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_OUTPUT_DIR = DEFAULT_CODEREVAL_ROOT / "agent_run_output"


def assert_target_file_is_clean(path: Path) -> None:
    source = path.read_text(encoding="utf-8", errors="ignore")
    ast.parse(source)


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
@dataclass
class TaskInfo:
    task_id: str
    name: str
    project: str
    package: str
    file_path: str
    lineno: int
    end_lineno: int
    code: str
    docstring: str
    human_label: str
    raw_task: Dict[str, Any]


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "Missing dependency: pyyaml. Please install with `pip install pyyaml`."
        ) from e

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Config root must be a dict.")
    return data


def to_int(value: Any, field_name: str) -> int:
    try:
        return int(value)
    except Exception as e:
        raise ValueError(f"Field `{field_name}` cannot be converted to int: {value!r}") from e


def iter_tasks(data: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
    elif isinstance(data, dict):
        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(value, dict):
                yield value


def build_task_info(task: Dict[str, Any]) -> TaskInfo:
    return TaskInfo(
        task_id=str(task.get("_id", "")),
        name=str(task.get("name", "")),
        project=str(task.get("project", "")),
        package=str(task.get("package", "")),
        file_path=str(task.get("file_path", "")),
        lineno=to_int(task.get("lineno"), "lineno"),
        end_lineno=to_int(task.get("end_lineno"), "end_lineno"),
        code=str(task.get("code", "")),
        docstring=str(task.get("docstring", "")),
        human_label=str(task.get("human_label", "")),
        raw_task=task,
    )


def find_task(
    json_path: Path,
    task_id: Optional[str] = None,
    task_name: Optional[str] = None,
) -> TaskInfo:
    data = load_json(json_path)
    tasks = list(iter_tasks(data))
    if not tasks:
        raise RuntimeError(f"No task objects found in {json_path}")

    for task in tasks:
        if task_id and str(task.get("_id", "")) == task_id:
            return build_task_info(task)
        if task_name and str(task.get("name", "")) == task_name:
            return build_task_info(task)

    raise RuntimeError(
        f"Task not found. task_id={task_id!r}, task_name={task_name!r}"
    )


def resolve_target_file(project_root: Path, file_path: str) -> Path:
    candidates = [
        project_root / file_path,
        project_root / "src" / file_path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Target file not found in either location:\n"
        + "\n".join(str(c) for c in candidates)
    )


def resolve_pipeline():
    try:
        from beacon_system.pipeline import run_pipeline  # type: ignore
        return run_pipeline
    except Exception as e:
        raise RuntimeError(f"Cannot import beacon_system.pipeline.run_pipeline: {e}") from e


# -----------------------------------------------------------------------------
# Config overlay
# -----------------------------------------------------------------------------
def overlay_model_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inject runtime model config from env if present.
    Supports both config['llm'] and config['model'] styles.
    """
    cfg = dict(config)

    llm_cfg = dict(cfg.get("llm") or {})
    model_cfg = dict(cfg.get("model") or {})

    env_base_url = os.environ.get("MODEL_BASE_URL")
    env_api_key = os.environ.get("MODEL_API_KEY")
    env_model_name = os.environ.get("MODEL_NAME")
    env_outputs_dir = os.environ.get("OUTPUTS_DIR")

    if env_base_url:
        llm_cfg.setdefault("base_url", env_base_url)
        model_cfg.setdefault("base_url", env_base_url)

    if env_api_key:
        llm_cfg["api_key"] = env_api_key
        model_cfg["api_key"] = env_api_key

    if env_model_name:
        llm_cfg.setdefault("model_name", env_model_name)
        model_cfg["name"] = env_model_name

    if llm_cfg:
        cfg["llm"] = llm_cfg
    if model_cfg:
        cfg["model"] = model_cfg

    if env_outputs_dir:
        run_cfg = dict(cfg.get("run") or {})
        run_cfg["outputs_dir"] = env_outputs_dir
        cfg["run"] = run_cfg

    return cfg


# -----------------------------------------------------------------------------
# Signature extraction
# -----------------------------------------------------------------------------
def extract_signature_from_code(code: str, expected_name: str) -> str:
    """
    Extract `def xxx(...):` signature from the original function body snippet.
    Falls back to a simple synthesized signature if parsing fails.
    """
    try:
        tree = ast.parse(code)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == expected_name:
                    try:
                        args_str = ast.unparse(node.args)
                    except Exception:
                        args_str = "*args, **kwargs"

                    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
                    ret = ""
                    if getattr(node, "returns", None) is not None:
                        try:
                            ret = f" -> {ast.unparse(node.returns)}"
                        except Exception:
                            ret = ""
                    return f"{prefix} {node.name}({args_str}){ret}:"
    except Exception:
        pass

    return f"def {expected_name}(*args, **kwargs):"


# -----------------------------------------------------------------------------
# Task translation: CoderEval -> Beacon task object
# -----------------------------------------------------------------------------
def build_beacon_task(task: TaskInfo, project_root: Path) -> Dict[str, Any]:
    target_file = resolve_target_file(project_root, task.file_path)
    signature = extract_signature_from_code(task.code, task.name)

    context_blocks = [
        f"CoderEval project: {task.project}",
        f"Package: {task.package}",
        f"Target file: {task.file_path}",
        f"Target line range: {task.lineno}-{task.end_lineno}",
        "Generate only the target function implementation compatible with surrounding project code.",
        "Do not introduce unrelated refactors.",
        "Prefer minimal, deterministic, project-compatible edits.",
        "Return only valid Python code.",
        "Do not include markdown fences.",
        "Do not include explanation text before or after the function.",
    ]

    if task.human_label:
        context_blocks.append(f"Human label: {task.human_label}")
    if task.docstring:
        context_blocks.append(f"Task docstring/comment: {task.docstring}")

    return {
        "task_id": task.task_id,
        "lang": "python",
        "entry_function": task.name,
        "signature": signature,
        "docstring": task.docstring or f"Implement function `{task.name}` correctly in project context.",
        "context_blocks": context_blocks,
        "runnable_level": "project_runnable",
        "project_context": {
            "project_root": str(project_root),
            "target_file": str(target_file.relative_to(project_root)),
            "original_file_path": str(target_file),
            "file_path": task.file_path,
            "replace_lineno": task.lineno,
            "replace_end_lineno": task.end_lineno,
            "package": task.package,
            "project": task.project,
            "constraints": [
                f"must_define:{task.name}",
                "must_be_python",
                "project_compatible",
                "keep_import_contract_if_needed",
                "no_unrelated_refactor",
                "no_markdown_fence",
                "code_only_output",
            ],
            "reference_original_candidate": task.code,
            "raw_task": task.raw_task,
        },
    }


# -----------------------------------------------------------------------------
# Artifacts
# -----------------------------------------------------------------------------
def write_task_artifacts(output_dir: Path, task: TaskInfo, beacon_task: Dict[str, Any]) -> Path:
    task_dir = output_dir / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "raw_task.json").write_text(
        json.dumps(task.raw_task, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (task_dir / "beacon_task.json").write_text(
        json.dumps(beacon_task, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (task_dir / "original_candidate.py").write_text(task.code, encoding="utf-8")
    return task_dir


# -----------------------------------------------------------------------------
# Candidate sanitation / validation
# -----------------------------------------------------------------------------
def sanitize_generated_code(code: str, entry_function: str) -> str:
    code = (code or "").strip()

    if code.startswith("```"):
        lines = code.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines).strip()

    markers = [f"def {entry_function}(", f"async def {entry_function}("]
    positions = [code.find(m) for m in markers if code.find(m) >= 0]
    if not positions:
        return code.strip()

    code = code[min(positions):].strip()

    lines = code.splitlines()
    kept = []
    base_indent = 0

    for i, line in enumerate(lines):
        stripped = line.strip()

        if i == 0:
            kept.append(line)
            base_indent = len(line) - len(line.lstrip(" "))
            continue

        current_indent = len(line) - len(line.lstrip(" "))

        if stripped == "":
            kept.append(line)
            continue

        if current_indent <= base_indent and not line.startswith((" ", "\t")):
            if stripped.startswith("def ") or stripped.startswith("async def ") or stripped.startswith("class "):
                break
            if not stripped.startswith("@"):
                break

        kept.append(line)

    return "\n".join(kept).rstrip() + "\n"


def validate_generated_code_syntax(generated_code: str, entry_function: str) -> None:
    tree = ast.parse(generated_code)

    found = False
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry_function:
            found = True
            break

    if not found:
        raise RuntimeError(f"Generated code does not define target function `{entry_function}`.")

    for node in tree.body:
        if isinstance(
            node,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Import,
                ast.ImportFrom,
                ast.ClassDef,
            ),
        ):
            continue

        if (
            isinstance(node, ast.Expr)
            and isinstance(getattr(node, "value", None), ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue

        raise RuntimeError(
            f"Generated code contains unsupported top-level node: {type(node).__name__}"
        )


def validate_full_python_file(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    ast.parse(source)


# -----------------------------------------------------------------------------
# Candidate injection / restore
# -----------------------------------------------------------------------------
def make_backup_path(target_file: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return target_file.with_name(f"{target_file.name}.{timestamp}.bak")


def inject_candidate_code(
    generated_code: str,
    task: TaskInfo,
    project_root: Path,
) -> tuple[Path, Path]:
    target_file = resolve_target_file(project_root, task.file_path)

    if task.lineno <= 0 or task.end_lineno < task.lineno:
        raise ValueError(
            f"Invalid line range: lineno={task.lineno}, end_lineno={task.end_lineno}"
        )

    original_text = target_file.read_text(encoding="utf-8")
    lines = original_text.splitlines()

    if task.end_lineno > len(lines):
        raise ValueError(
            f"end_lineno={task.end_lineno} exceeds file length={len(lines)} "
            f"for {target_file}"
        )

    backup_path = make_backup_path(target_file)
    shutil.copy2(target_file, backup_path)

    new_lines = (
        lines[: task.lineno - 1]
        + generated_code.splitlines()
        + lines[task.end_lineno :]
    )
    new_text = "\n".join(new_lines) + "\n"
    target_file.write_text(new_text, encoding="utf-8")

    return target_file, backup_path


def restore_backup(target_file: Path, backup_path: Path) -> None:
    if backup_path.exists():
        target_file.write_text(
            backup_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )


# -----------------------------------------------------------------------------
# Pytest runner
# -----------------------------------------------------------------------------
def run_pytest(
    project_root: Path,
    pytest_target: str,
    pytest_k: Optional[str] = None,
    set_pythonpath_src: bool = False,
) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, "-m", "pytest", pytest_target, "-q"]
    if pytest_k:
        cmd.extend(["-k", pytest_k])

    env = os.environ.copy()

    if set_pythonpath_src:
        src_path = str(project_root / "src")
        existing = env.get("PYTHONPATH", "")
        sep = ";" if os.name == "nt" else ":"
        env["PYTHONPATH"] = src_path if not existing else f"{src_path}{sep}{existing}"

    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=env,
        text=True,
        capture_output=True,
    )
    return result


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one CoderEval task through Beacon agent and validate with pytest."
    )
    parser.add_argument(
        "--task-id",
        default="62e60f43d76274f8a4026e28",
        help="CoderEval task _id",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Fallback task name, e.g. hydrate_time",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="Path to CoderEval4Python.json",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=DEFAULT_PROJECT_ROOT,
        help="Local cloned project root",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Beacon config path",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Artifact output directory",
    )
    parser.add_argument(
        "--pytest-target",
        default=r"tests\unit\common\codec\hydration\v1\test_temporal_hydration.py",
        help="Pytest file / nodeid to run",
    )
    parser.add_argument(
        "--pytest-k",
        default="hydrate_time",
        help="Pytest -k expression",
    )
    parser.add_argument(
        "--set-pythonpath-src",
        action="store_true",
        help="Set PYTHONPATH=<project_root>/src before running pytest",
    )

    args = parser.parse_args()

    print("=" * 100)
    print("CoderEval single-task Beacon-agent runner")
    print(f"repo_root      : {REPO_ROOT}")
    print(f"config_path    : {args.config_path}")
    print(f"json_path      : {args.json_path}")
    print(f"project_root   : {args.project_root}")
    print(f"pytest_target  : {args.pytest_target}")
    print(f"pytest_k       : {args.pytest_k}")
    print("=" * 100)

    try:
        task = find_task(args.json_path, task_id=args.task_id, task_name=args.task_name)
    except Exception as e:
        print(f"[FAIL] find_task failed: {e}")
        traceback.print_exc()
        return 1

    print(f"[OK] Task found      : {task.task_id} / {task.name}")
    print(f"[OK] Project         : {task.project}")
    print(f"[OK] File path       : {task.file_path}")
    print(f"[OK] Line range      : {task.lineno}-{task.end_lineno}")

    try:
        config = load_yaml(args.config_path)
        config = overlay_model_env(config)
    except Exception as e:
        print(f"[FAIL] config load failed: {e}")
        traceback.print_exc()
        return 2

    try:
        beacon_task = build_beacon_task(task, args.project_root)
    except Exception as e:
        print(f"[FAIL] build_beacon_task failed: {e}")
        traceback.print_exc()
        return 3

    print("[OK] Beacon task:")
    print(_pretty(beacon_task))

    task_dir = write_task_artifacts(args.output_dir, task, beacon_task)
    print(f"[OK] Artifacts saved : {task_dir}")

    try:
        run_pipeline = resolve_pipeline()
    except Exception as e:
        print(f"[FAIL] pipeline import failed: {e}")
        traceback.print_exc()
        return 4

    try:
        result = run_pipeline(task=beacon_task, config=config)
    except Exception as e:
        print(f"[FAIL] pipeline execution failed: {e}")
        traceback.print_exc()
        return 5

    print("[OK] Pipeline result:")
    print(_pretty(result))

    (task_dir / "pipeline_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    run_dir_raw = result.get("run_dir")
    if not run_dir_raw:
        print("[FAIL] pipeline result missing run_dir")
        return 6

    run_dir = Path(run_dir_raw)
    code_path = run_dir / "code_round1.py"
    generation_info_path = run_dir / "generation_round1.json"
    verifier_info_path = run_dir / "verifier_round1.json"
    exec_info_path = run_dir / "exec_round1.json"

    if generation_info_path.exists():
        print("[INFO] generation_round1.json found")
    if verifier_info_path.exists():
        print("[INFO] verifier_round1.json found")
    if exec_info_path.exists():
        print("[INFO] exec_round1.json found")

    pipeline_status = result.get("status")
    if pipeline_status != "ok":
        print(f"[FAIL] pipeline status is not ok: {pipeline_status}")
        print(f"[INFO] error_stage: {result.get('error_stage')}")
        print(f"[INFO] error      : {result.get('error')}")
        return 10

    if not code_path.exists():
        print(f"[FAIL] generated code not found: {code_path}")
        return 7

    generated_code = code_path.read_text(encoding="utf-8", errors="ignore")
    if not generated_code.strip():
        print("[FAIL] generated code is empty")
        return 8

    (task_dir / "agent_generated_code_raw.py").write_text(
        generated_code,
        encoding="utf-8",
    )

    generated_code = sanitize_generated_code(generated_code, task.name)

    (task_dir / "agent_generated_code_sanitized.py").write_text(
        generated_code,
        encoding="utf-8",
    )

    try:
        validate_generated_code_syntax(generated_code, task.name)
    except Exception as e:
        print(f"[FAIL] generated code validation failed: {e}")
        traceback.print_exc()
        return 9

    print(f"[OK] Generated code ready: {code_path}")

    target_file: Optional[Path] = None
    backup_path: Optional[Path] = None

    try:
        target_file, backup_path = inject_candidate_code(
            generated_code=generated_code,
            task=task,
            project_root=args.project_root,
        )
        print(f"[OK] Injected into   : {target_file}")
        print(f"[OK] Backup path     : {backup_path}")

        try:
            validate_full_python_file(target_file)
            print(f"[OK] Full file syntax check passed: {target_file}")
        except Exception as e:
            print(f"[FAIL] Injected file syntax invalid: {e}")
            traceback.print_exc()
            return 11

        pytest_result = run_pytest(
            project_root=args.project_root,
            pytest_target=args.pytest_target,
            pytest_k=args.pytest_k,
            set_pythonpath_src=args.set_pythonpath_src,
        )

        (task_dir / "pytest_stdout.txt").write_text(
            pytest_result.stdout or "",
            encoding="utf-8",
        )
        (task_dir / "pytest_stderr.txt").write_text(
            pytest_result.stderr or "",
            encoding="utf-8",
        )
        (task_dir / "pytest_meta.json").write_text(
            json.dumps(
                {
                    "returncode": pytest_result.returncode,
                    "pytest_target": args.pytest_target,
                    "pytest_k": args.pytest_k,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print("-" * 100)
        print("[PYTEST STDOUT]")
        print(pytest_result.stdout.strip() or "<empty>")
        print("-" * 100)
        print("[PYTEST STDERR]")
        print(pytest_result.stderr.strip() or "<empty>")
        print("-" * 100)

        if pytest_result.returncode == 0:
            print("[PASS] Agent + CoderEval single-task run succeeded.")
            print(f"[PASS] run_dir        : {run_dir}")
            print(f"[PASS] task_artifacts : {task_dir}")
            return 0

        print(f"[FAIL] Pytest failed with return code {pytest_result.returncode}.")
        print(f"[INFO] run_dir        : {run_dir}")
        print(f"[INFO] task_artifacts : {task_dir}")
        return pytest_result.returncode

    finally:
        if target_file is not None and backup_path is not None and backup_path.exists():
            restore_backup(target_file, backup_path)
            print(f"[OK] Restored original file from: {backup_path}")


if __name__ == "__main__":
    raise SystemExit(main())