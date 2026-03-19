# src/debug_codereval_injection_rootcause.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import ast
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_CODEREVAL_ROOT = REPO_ROOT / "benchmarks" / "CoderEval"
DEFAULT_JSON_PATH = DEFAULT_CODEREVAL_ROOT / "CoderEval4Python.json"
DEFAULT_WORKSPACE = DEFAULT_CODEREVAL_ROOT / "workspace"
DEFAULT_PROJECT_ROOT = DEFAULT_WORKSPACE / "neo4j-python-driver"
DEFAULT_OUTPUT_DIR = DEFAULT_CODEREVAL_ROOT / "injection_debug_output"


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


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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

    raise RuntimeError(f"Task not found. task_id={task_id!r}, task_name={task_name!r}")


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


def parse_python_or_error(source: str) -> tuple[bool, Optional[str]]:
    try:
        ast.parse(source)
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def find_function_ast_range(source: str, func_name: str) -> tuple[int, int]:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
                raise RuntimeError("AST node missing lineno/end_lineno; Python 3.8+ required.")
            return int(node.lineno), int(node.end_lineno)
    raise RuntimeError(f"Function `{func_name}` not found in module AST.")


def replace_by_line_range(source: str, start: int, end: int, new_code: str) -> str:
    lines = source.splitlines()
    if start <= 0 or end < start:
        raise ValueError(f"Invalid replace range: {start}-{end}")
    if end > len(lines):
        raise ValueError(f"Replace end {end} exceeds file length {len(lines)}")

    new_lines = lines[: start - 1] + new_code.splitlines() + lines[end :]
    return "\n".join(new_lines) + "\n"


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def preview_lines(source: str, start: int, end: int, context: int = 5) -> str:
    lines = source.splitlines()
    lo = max(1, start - context)
    hi = min(len(lines), end + context)
    out = []
    for i in range(lo, hi + 1):
        marker = ">>" if start <= i <= end else "  "
        out.append(f"{marker} {i:04d}: {lines[i - 1]}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Diagnose whether syntax errors come from generated code or injection strategy."
    )
    parser.add_argument(
        "--task-id",
        default="62e60f43d76274f8a4026e28",
        help="CoderEval task _id",
    )
    parser.add_argument(
        "--task-name",
        default=None,
        help="Fallback task name",
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
        "--generated-code",
        type=Path,
        required=True,
        help="Path to generated function code, e.g. outputs/runs/.../code_round1.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save debug outputs",
    )
    args = parser.parse_args()

    print("=" * 100)
    print("CoderEval injection root-cause debugger")
    print(f"repo_root      : {REPO_ROOT}")
    print(f"json_path      : {args.json_path}")
    print(f"project_root   : {args.project_root}")
    print(f"generated_code : {args.generated_code}")
    print(f"output_dir     : {args.output_dir}")
    print("=" * 100)

    try:
        task = find_task(args.json_path, task_id=args.task_id, task_name=args.task_name)
    except Exception as e:
        print(f"[FAIL] find_task failed: {e}")
        traceback.print_exc()
        return 1

    try:
        target_file = resolve_target_file(args.project_root, task.file_path)
    except Exception as e:
        print(f"[FAIL] resolve_target_file failed: {e}")
        traceback.print_exc()
        return 2

    if not args.generated_code.exists():
        print(f"[FAIL] generated code file not found: {args.generated_code}")
        return 3

    task_out = args.output_dir / task.task_id
    task_out.mkdir(parents=True, exist_ok=True)

    original_source = target_file.read_text(encoding="utf-8", errors="ignore")
    generated_code = args.generated_code.read_text(encoding="utf-8", errors="ignore").strip() + "\n"

    save_text(task_out / "original_target_file.py", original_source)
    save_text(task_out / "generated_code.py", generated_code)

    dataset_start, dataset_end = task.lineno, task.end_lineno
    print(f"[INFO] dataset range : {dataset_start}-{dataset_end}")

    try:
        ast_start, ast_end = find_function_ast_range(original_source, task.name)
        print(f"[INFO] ast range     : {ast_start}-{ast_end}")
    except Exception as e:
        print(f"[FAIL] AST function range detection failed: {e}")
        traceback.print_exc()
        return 4

    generated_ok, generated_err = parse_python_or_error(generated_code)
    print(f"[CHECK] generated code parse ok : {generated_ok}")
    if generated_err:
        print(f"[CHECK] generated code parse err: {generated_err}")

    original_ok, original_err = parse_python_or_error(original_source)
    print(f"[CHECK] original file parse ok   : {original_ok}")
    if original_err:
        print(f"[CHECK] original file parse err  : {original_err}")

    dataset_preview = preview_lines(original_source, dataset_start, dataset_end, context=5)
    ast_preview = preview_lines(original_source, ast_start, ast_end, context=5)

    save_text(task_out / "dataset_range_preview.txt", dataset_preview)
    save_text(task_out / "ast_range_preview.txt", ast_preview)

    print("-" * 100)
    print("[DATASET RANGE PREVIEW]")
    print(dataset_preview)
    print("-" * 100)
    print("[AST RANGE PREVIEW]")
    print(ast_preview)
    print("-" * 100)

    diagnosis: Dict[str, Any] = {
        "task_id": task.task_id,
        "task_name": task.name,
        "target_file": str(target_file),
        "dataset_range": [dataset_start, dataset_end],
        "ast_range": [ast_start, ast_end],
        "generated_code_parse_ok": generated_ok,
        "generated_code_parse_error": generated_err,
        "original_file_parse_ok": original_ok,
        "original_file_parse_error": original_err,
        "dataset_vs_ast_same": [dataset_start, dataset_end] == [ast_start, ast_end],
    }

    # Variant 1: replace by dataset line range
    try:
        by_dataset = replace_by_line_range(original_source, dataset_start, dataset_end, generated_code)
        save_text(task_out / "injected_by_dataset_range.py", by_dataset)
        ok1, err1 = parse_python_or_error(by_dataset)
        diagnosis["injected_by_dataset_range_parse_ok"] = ok1
        diagnosis["injected_by_dataset_range_parse_error"] = err1
        print(f"[CHECK] injected by dataset range parse ok : {ok1}")
        if err1:
            print(f"[CHECK] injected by dataset range err    : {err1}")
    except Exception as e:
        diagnosis["injected_by_dataset_range_parse_ok"] = False
        diagnosis["injected_by_dataset_range_parse_error"] = repr(e)
        print(f"[CHECK] injected by dataset range failed   : {e}")

    # Variant 2: replace by AST function range
    try:
        by_ast = replace_by_line_range(original_source, ast_start, ast_end, generated_code)
        save_text(task_out / "injected_by_ast_range.py", by_ast)
        ok2, err2 = parse_python_or_error(by_ast)
        diagnosis["injected_by_ast_range_parse_ok"] = ok2
        diagnosis["injected_by_ast_range_parse_error"] = err2
        print(f"[CHECK] injected by AST range parse ok     : {ok2}")
        if err2:
            print(f"[CHECK] injected by AST range err         : {err2}")
    except Exception as e:
        diagnosis["injected_by_ast_range_parse_ok"] = False
        diagnosis["injected_by_ast_range_parse_error"] = repr(e)
        print(f"[CHECK] injected by AST range failed       : {e}")

    # Heuristic interpretation
    conclusion = []
    if not generated_ok:
        conclusion.append("生成代码片段本身就是非法 Python，根因更偏向生成模块。")
    else:
        if diagnosis.get("injected_by_dataset_range_parse_ok") is False and diagnosis.get("injected_by_ast_range_parse_ok") is True:
            conclusion.append("数据集 line range 替换会破坏文件，但 AST range 替换不会，根因是数据集行号/文本切片替换不稳定。")
        elif diagnosis.get("injected_by_dataset_range_parse_ok") is False and diagnosis.get("injected_by_ast_range_parse_ok") is False:
            conclusion.append("无论数据集 range 还是 AST range，替换后整文件都非法，根因更偏向生成函数与上下文不兼容。")
        elif diagnosis.get("injected_by_dataset_range_parse_ok") is True and diagnosis.get("injected_by_ast_range_parse_ok") is True:
            conclusion.append("两种替换都能通过整文件 parse，SyntaxError 可能来自你实际注入逻辑与此诊断脚本逻辑不一致。")
        elif diagnosis.get("injected_by_dataset_range_parse_ok") is True and diagnosis.get("injected_by_ast_range_parse_ok") is False:
            conclusion.append("AST range 替换反而失败，说明目标定位或函数结构存在特殊情况，需要检查模块内真实函数边界。")

    diagnosis["conclusion"] = conclusion

    save_text(task_out / "diagnosis.json", _pretty(diagnosis))

    print("=" * 100)
    print("[DIAGNOSIS]")
    print(_pretty(diagnosis))
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())