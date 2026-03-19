# scripts/test_agent_system_smoke.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


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


def _build_example_task() -> Dict[str, Any]:
    """
    Keep the task minimal but enough to exercise:
    task ingest -> reader/ir -> constraints -> generator -> verifier -> artifacts
    """
    return {
        "task_id": "smoke_agent_001",
        "lang": "python",
        "entry_function": "add",
        "signature": "def add(a: int, b: int) -> int:",
        "docstring": "Return the sum of two integers.",
        "context_blocks": [
            "Return only deterministic logic.",
            "Do not read files.",
            "Do not use network.",
            "Keep the implementation minimal.",
        ],
        "runnable_level": "unit_runnable",
        "project_context": {
            "target_file": "generated_add.py",
            "constraints": [
                "must_define:add",
                "must_be_python",
                "deterministic_only",
                "no_network",
            ],
            "tests": [
                {
                    "name": "basic_positive",
                    "call": "add(1, 2)",
                    "expected": 3,
                },
                {
                    "name": "basic_mixed",
                    "call": "add(-1, 1)",
                    "expected": 0,
                },
            ],
        },
    }


def _resolve_pipeline_api() -> Tuple[Optional[Any], Optional[Any]]:
    """
    Try a few public API shapes without forcing one exact contract.
    Returns (run_pipeline_fn, Pipeline_class).
    """
    run_pipeline = None
    Pipeline = None

    try:
        from beacon_system.pipeline import run_pipeline as _rp  # type: ignore
        run_pipeline = _rp
    except Exception:
        pass

    try:
        from beacon_system.pipeline import Pipeline as _pl  # type: ignore
        Pipeline = _pl
    except Exception:
        pass

    return run_pipeline, Pipeline


def _invoke_pipeline(task: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    run_pipeline, Pipeline = _resolve_pipeline_api()

    if run_pipeline is None and Pipeline is None:
        raise RuntimeError(
            "Cannot find a public pipeline entry.\n"
            "Expected one of:\n"
            "  - beacon_system.pipeline.run_pipeline\n"
            "  - beacon_system.pipeline.Pipeline"
        )

    last_error: Optional[Exception] = None

    if run_pipeline is not None:
        for mode in ("kw", "pos"):
            try:
                if mode == "kw":
                    result = run_pipeline(task=task, config=config)
                else:
                    result = run_pipeline(task, config)
                if isinstance(result, dict):
                    return result
                return {"status": "ok", "result": result}
            except Exception as e:
                last_error = e

    if Pipeline is not None:
        # Try a few constructor/run shapes
        candidates = [
            ("init_config__run_task", lambda: Pipeline(config=config).run(task)),
            ("init_empty__run_task_config_kw", lambda: Pipeline().run(task, config=config)),
            ("init_empty__run_task_only", lambda: Pipeline().run(task)),
            ("init_config_only__run_noargs", lambda: Pipeline(config=config).run()),
        ]
        for _, fn in candidates:
            try:
                result = fn()
                if isinstance(result, dict):
                    return result
                return {"status": "ok", "result": result}
            except Exception as e:
                last_error = e

    raise RuntimeError(f"Pipeline invocation failed. Last error: {last_error}")


def _guess_run_dir(result: Dict[str, Any]) -> Optional[Path]:
    candidates = [
        result.get("run_dir"),
        result.get("output_dir"),
        result.get("artifacts_dir"),
    ]

    artifacts = result.get("artifacts")
    if isinstance(artifacts, dict):
        for v in artifacts.values():
            if isinstance(v, str):
                p = Path(v)
                if p.exists():
                    return p.parent

    for c in candidates:
        if isinstance(c, str) and c.strip():
            p = Path(c)
            if p.exists():
                return p

    # fallback: newest dir under outputs/runs
    runs_root = REPO_ROOT / "outputs" / "runs"
    if runs_root.exists():
        dirs = [p for p in runs_root.iterdir() if p.is_dir()]
        if dirs:
            dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return dirs[0]

    return None


def _check_run_artifacts(run_dir: Path) -> Dict[str, bool]:
    checks = {
        "config.yaml": (run_dir / "config.yaml").exists(),
        "task.json": (run_dir / "task.json").exists(),
        "ir.json": (run_dir / "ir.json").exists(),
        "constraints.json": (run_dir / "constraints.json").exists(),
        "code_round1.py": (run_dir / "code_round1.py").exists(),
        "verifier_round1.json": (run_dir / "verifier_round1.json").exists(),
    }

    # execution artifacts are optional in the first smoke run
    checks["exec_round1.json"] = (run_dir / "exec_round1.json").exists()
    checks["adapter_snapshot.json"] = (run_dir / "adapter_snapshot.json").exists()
    return checks


def _read_text_if_exists(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


def _soft_validate_generated_code(run_dir: Path) -> None:
    code_path = run_dir / "code_round1.py"
    if not code_path.exists():
        return

    import ast

    code = _read_text_if_exists(code_path) or ""
    ast.parse(code)

    namespace: Dict[str, Any] = {}
    exec(code, namespace)

    if "add" in namespace and callable(namespace["add"]):
        assert namespace["add"](1, 2) == 3
        assert namespace["add"](-1, 1) == 0


def main() -> int:
    print("=" * 80)
    print("[Smoke] Beacon agent system smoke test")
    print(f"[Smoke] repo_root   = {REPO_ROOT}")

    config_path = REPO_ROOT / "configs" / "default.yaml"
    print(f"[Smoke] config_path = {config_path}")

    try:
        config = _load_yaml(config_path)
        print("[Smoke] config loaded.")
    except Exception as e:
        print(f"[FAIL] config load failed: {e}")
        traceback.print_exc()
        return 1

    task = _build_example_task()
    print("[Smoke] task prepared:")
    print(_pretty(task))

    try:
        result = _invoke_pipeline(task, config)
        print("[Smoke] pipeline returned:")
        print(_pretty(result))
    except Exception as e:
        print(f"[FAIL] pipeline invocation failed: {e}")
        traceback.print_exc()
        return 2

    run_dir = _guess_run_dir(result)
    if not run_dir:
        print("[FAIL] cannot resolve run_dir from pipeline result or outputs/runs.")
        return 3

    print(f"[Smoke] resolved run_dir = {run_dir}")

    checks = _check_run_artifacts(run_dir)
    for name, ok in checks.items():
        print(f"[Check] exists {name}: {ok}")

    required = [
        "task.json",
        "ir.json",
        "constraints.json",
        "code_round1.py",
        "verifier_round1.json",
    ]
    missing = [name for name in required if not checks.get(name, False)]
    if missing:
        print(f"[FAIL] missing required artifacts: {missing}")
        return 4

    try:
        _soft_validate_generated_code(run_dir)
        print("[Check] generated code soft validation: True")
    except Exception as e:
        print(f"[FAIL] generated code soft validation failed: {e}")
        traceback.print_exc()
        return 5

    print("[Smoke] AGENT_SYSTEM_OK")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())