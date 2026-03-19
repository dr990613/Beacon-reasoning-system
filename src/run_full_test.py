# src/run_full_test.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _pretty(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("Missing dependency: pyyaml. Please install with `pip install pyyaml`.") from e

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Config root must be a dict.")
    return data


def _build_full_example_task() -> Dict[str, Any]:
    """
    A stricter end-to-end task than the previous smoke example.
    It verifies:
    - task ingestion
    - prompt generation
    - code generation
    - artifact writing
    - executable behavior
    """
    return {
        "task_id": "full_e2e_agent_001",
        "lang": "python",
        "entry_function": "add",
        "signature": "def add(a: int, b: int) -> int:",
        "docstring": (
            "Return the sum of two integers. "
            "The implementation must be deterministic and minimal."
        ),
        "context_blocks": [
            "Return only deterministic logic.",
            "Do not read files.",
            "Do not use network.",
            "Do not import unnecessary libraries.",
            "Keep the implementation minimal and correct.",
        ],
        "runnable_level": "unit_runnable",
        "project_context": {
            "target_file": "generated_add.py",
            "constraints": [
                "must_define:add",
                "must_be_python",
                "deterministic_only",
                "no_network",
                "no_file_io",
            ],
            "tests": [
                {"name": "basic_positive", "call": "add(1, 2)", "expected": 3},
                {"name": "basic_zero", "call": "add(0, 0)", "expected": 0},
                {"name": "basic_mixed", "call": "add(-1, 1)", "expected": 0},
                {"name": "basic_negative", "call": "add(-3, -4)", "expected": -7},
            ],
        },
    }


def _resolve_pipeline():
    try:
        from beacon_system.pipeline import run_pipeline  # type: ignore
        return run_pipeline
    except Exception as e:
        raise RuntimeError(f"Cannot import beacon_system.pipeline.run_pipeline: {e}") from e


def _read_text_if_exists(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_json_if_exists(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_required_artifacts(run_dir: Path) -> Dict[str, bool]:
    checks = {
        "config.yaml": (run_dir / "config.yaml").exists(),
        "task.json": (run_dir / "task.json").exists(),
        "ir.json": (run_dir / "ir.json").exists(),
        "constraints.json": (run_dir / "constraints.json").exists(),
        "code_round1.py": (run_dir / "code_round1.py").exists(),
        "verifier_round1.json": (run_dir / "verifier_round1.json").exists(),
        "exec_round1.json": (run_dir / "exec_round1.json").exists(),
        "adapter_snapshot.json": (run_dir / "adapter_snapshot.json").exists(),
    }
    return checks


def _validate_generated_code(code: str, entry_function: str, tests: list[dict]) -> Dict[str, Any]:
    """
    Strict post-check:
    1) syntax parse
    2) exec
    3) function exists
    4) task tests pass
    """
    result: Dict[str, Any] = {
        "syntax_ok": False,
        "exec_ok": False,
        "entry_ok": False,
        "passed": 0,
        "failed": 0,
        "details": [],
    }

    ast.parse(code)
    result["syntax_ok"] = True

    namespace: Dict[str, Any] = {}
    exec(code, namespace)
    result["exec_ok"] = True

    if entry_function not in namespace or not callable(namespace[entry_function]):
        raise RuntimeError(f"Missing callable entry function: {entry_function}")
    result["entry_ok"] = True

    for t in tests:
        call_expr = str(t["call"])
        expected = t["expected"]
        name = t["name"]

        actual = eval(call_expr, namespace)
        ok = (actual == expected)
        result["details"].append(
            {
                "name": name,
                "call": call_expr,
                "expected": expected,
                "actual": actual,
                "ok": ok,
            }
        )
        if ok:
            result["passed"] += 1
        else:
            result["failed"] += 1

    return result


def main() -> int:
    print("=" * 80)
    print("[FullTest] Beacon agent full end-to-end test")
    print(f"[FullTest] repo_root   = {REPO_ROOT}")

    config_path = REPO_ROOT / "configs" / "default.yaml"
    print(f"[FullTest] config_path = {config_path}")

    try:
        config = _load_yaml(config_path)
        print("[FullTest] config loaded.")
    except Exception as e:
        print(f"[FAIL] config load failed: {e}")
        traceback.print_exc()
        return 1

    task = _build_full_example_task()
    print("[FullTest] task prepared:")
    print(_pretty(task))

    try:
        run_pipeline = _resolve_pipeline()
    except Exception as e:
        print(f"[FAIL] pipeline import failed: {e}")
        traceback.print_exc()
        return 2

    try:
        result = run_pipeline(task=task, config=config)
    except Exception as e:
        print(f"[FAIL] pipeline execution failed: {e}")
        traceback.print_exc()
        return 3

    print("[FullTest] pipeline result:")
    print(_pretty(result))

    run_dir_raw = result.get("run_dir")
    if not run_dir_raw:
        print("[FAIL] pipeline result missing run_dir")
        return 4

    run_dir = Path(run_dir_raw)
    print(f"[FullTest] run_dir = {run_dir}")

    checks = _assert_required_artifacts(run_dir)
    for name, ok in checks.items():
        print(f"[Check] exists {name}: {ok}")

    missing = [name for name, ok in checks.items() if not ok]
    if missing:
        print(f"[FAIL] missing required artifacts: {missing}")
        return 5

    code = _read_text_if_exists(run_dir / "code_round1.py")
    if not code:
        print("[FAIL] code_round1.py is empty or unreadable")
        return 6

    try:
        validation = _validate_generated_code(
            code=code,
            entry_function=task["entry_function"],
            tests=task["project_context"]["tests"],
        )
    except Exception as e:
        print(f"[FAIL] generated code validation failed: {e}")
        traceback.print_exc()
        return 7

    print("[FullTest] generated code validation:")
    print(_pretty(validation))

    if validation["failed"] > 0:
        print("[FAIL] generated code did not pass all task tests")
        return 8

    verifier_payload = _read_json_if_exists(run_dir / "verifier_round1.json") or {}
    exec_payload = _read_json_if_exists(run_dir / "exec_round1.json") or {}

    print("[FullTest] verifier_round1.json:")
    print(_pretty(verifier_payload))

    print("[FullTest] exec_round1.json:")
    print(_pretty(exec_payload))

    if result.get("status") != "ok":
        print("[FAIL] pipeline status is not ok")
        return 9

    if result.get("exec_status") not in ("pass", "ok", "success"):
        print(f"[FAIL] exec_status is unexpected: {result.get('exec_status')}")
        return 10

    print("[FullTest] AGENT_FULL_E2E_OK")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())