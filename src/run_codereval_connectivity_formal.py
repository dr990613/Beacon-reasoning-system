# src/run_codereval_connectivity_formal.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

try:
    import yaml
except ImportError as e:
    print("[FATAL] Missing dependency: pyyaml")
    raise

# -----------------------------------------------------------------------------
# Path bootstrap
# -----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

DEFAULT_CODEREVAL_ROOT = REPO_ROOT / "benchmarks" / "CoderEval"
DEFAULT_JSON_PATH = DEFAULT_CODEREVAL_ROOT / "CoderEval4Python.json"
DEFAULT_PROJECT_ROOT = DEFAULT_CODEREVAL_ROOT / "workspace" / "neo4j-python-driver"
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.yaml"
DEFAULT_OUTPUT_DIR = DEFAULT_CODEREVAL_ROOT / "formal_connectivity_output"


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------
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


def overlay_model_env(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Inject runtime model config from env if present.
    Align with pipeline._build_model_config_from_dict.
    """
    cfg = dict(config)

    llm_cfg = dict(cfg.get("llm") or {})
    model_cfg = dict(cfg.get("model") or {})
    run_cfg = dict(cfg.get("run") or {})

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

    if env_outputs_dir:
        run_cfg["outputs_dir"] = env_outputs_dir

    if llm_cfg:
        cfg["llm"] = llm_cfg
    if model_cfg:
        cfg["model"] = model_cfg
    if run_cfg:
        cfg["run"] = run_cfg

    return cfg


def iter_tasks(data: Any):
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item
        return

    if isinstance(data, dict):
        if "RECORDS" in data and isinstance(data["RECORDS"], list):
            for item in data["RECORDS"]:
                if isinstance(item, dict):
                    yield item
            return

        for value in data.values():
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        yield item
            elif isinstance(value, dict):
                yield value


def select_first_n_tasks(json_path: Path, n: int) -> List[Dict[str, Any]]:
    data = load_json(json_path)
    tasks = list(iter_tasks(data))
    if not tasks:
        raise RuntimeError(f"No task objects found in {json_path}")
    return tasks[:n]


def build_summary_row(
    *,
    task_index: int,
    raw_task: Dict[str, Any],
    result: Dict[str, Any],
    error: str | None = None,
) -> Dict[str, Any]:
    task_id = str(raw_task.get("_id") or raw_task.get("task_id") or raw_task.get("id") or "")
    task_name = str(raw_task.get("name") or raw_task.get("function_name") or raw_task.get("method_name") or "")
    file_path = str(raw_task.get("file_path") or raw_task.get("file") or "")

    return {
        "task_index": task_index,
        "task_id": task_id,
        "task_name": task_name,
        "file_path": file_path,
        "status": result.get("status"),
        "error_stage": result.get("error_stage"),
        "error": result.get("error") if error is None else error,
        "verifier_ok": result.get("verifier_ok"),
        "exec_status": result.get("exec_status"),
        "rounds": result.get("rounds"),
        "run_dir": result.get("run_dir"),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_model_config_from_experiment_config(config: Dict[str, Any]):
    from beacon_system.llm.config import ModelConfig  # type: ignore

    llm_dict = dict(config.get("llm") or {})
    model_dict = dict(config.get("model") or {})

    merged: Dict[str, Any] = {}
    merged.update(model_dict)
    merged.update(llm_dict)

    def pick(*keys: str, default: Any = "") -> Any:
        for k in keys:
            if k in merged and merged[k] not in (None, ""):
                return merged[k]
            if k in config and config[k] not in (None, ""):
                return config[k]
        return default

    base_url = str(
        pick("base_url", "endpoint", "url")
        or os.getenv("MODEL_BASE_URL")
        or os.getenv("OPENAI_BASE_URL")
        or ""
    ).strip()

    api_key = str(
        pick("api_key", "key", "token")
        or os.getenv("MODEL_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or ""
    ).strip()

    model_name = str(
        pick("model_name", "model", "name")
        or os.getenv("MODEL_NAME")
        or ""
    ).strip()

    timeout_s = int(pick("timeout_s", default=120) or 120)

    params = merged.get("params")
    if not isinstance(params, dict):
        params = {
            "temperature": 0.2,
            "top_p": 0.95,
            "max_tokens": 2048,
        }

    if not base_url:
        raise ValueError("Missing model base_url.")
    if not api_key:
        raise ValueError("Missing model api_key.")
    if not model_name:
        raise ValueError("Missing model_name.")

    return ModelConfig(
        base_url=base_url,
        api_key=api_key,
        model_name=model_name,
        timeout_s=timeout_s,
        params=params,
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Formal connectivity experiment for Beacon agent + CoderEval (first 3 tasks)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--json-path", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-tasks", type=int, default=3)
    parser.add_argument(
        "--docker-image",
        required=True,
        help="Docker image for formal CoderEval evaluation.",
    )
    parser.add_argument(
        "--eval-cmd",
        default="pytest -q",
        help="Benchmark-aligned eval command to run inside Docker.",
    )
    args = parser.parse_args()

    print("=" * 100)
    print("Formal connectivity experiment: Beacon agent + CoderEval")
    print(f"config        : {args.config}")
    print(f"json_path     : {args.json_path}")
    print(f"project_root  : {args.project_root}")
    print(f"output_dir    : {args.output_dir}")
    print(f"num_tasks     : {args.num_tasks}")
    print(f"docker_image  : {args.docker_image}")
    print(f"eval_cmd      : {args.eval_cmd}")
    print("=" * 100)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = load_yaml(args.config)
        config = overlay_model_env(config)
    except Exception as e:
        print(f"[FATAL] config load failed: {e}")
        traceback.print_exc()
        return 1

    try:
        tasks = select_first_n_tasks(args.json_path, args.num_tasks)
    except Exception as e:
        print(f"[FATAL] task selection failed: {e}")
        traceback.print_exc()
        return 2

    print(f"[OK] Selected {len(tasks)} tasks for connectivity experiment.")

    try:
        from beacon_system.pipeline import run  # type: ignore
        from beacon_system.llm.client import LLMClient  # type: ignore
        from beacon_system.adapters.codereval_task_adapter import CoderEvalTaskAdapter  # type: ignore
        from beacon_system.adapters.codereval_docker_runtime_adapter import CoderEvalDockerRuntimeAdapter  # type: ignore
    except Exception as e:
        print(f"[FATAL] formal components import failed: {e}")
        traceback.print_exc()
        return 3

    try:
        llm_cfg = build_model_config_from_experiment_config(config)
        llm = LLMClient(llm_cfg)
    except Exception as e:
        print(f"[FATAL] LLM config/build failed: {e}")
        traceback.print_exc()
        return 4

    summary_rows: List[Dict[str, Any]] = []
    detailed_results: List[Dict[str, Any]] = []

    for idx, raw_task in enumerate(tasks):
        task_id = str(raw_task.get("_id") or raw_task.get("task_id") or raw_task.get("id") or f"task_{idx}")
        print("\n" + "#" * 100)
        print(f"[RUN] Connectivity task {idx + 1}/{len(tasks)} : {task_id}")
        print("#" * 100)

        task_result_dir = args.output_dir / f"{idx:02d}_{task_id}"
        task_result_dir.mkdir(parents=True, exist_ok=True)

        write_json(task_result_dir / "raw_task.json", raw_task)

        try:
            task_adapter = CoderEvalTaskAdapter(
                raw_task=raw_task,
                json_path=args.json_path,
                project_root=args.project_root,
            )

            runtime_adapter = CoderEvalDockerRuntimeAdapter(
                project_root=args.project_root,
                docker_image=args.docker_image,
                eval_cmd=args.eval_cmd,
                task_result_dir=task_result_dir,
            )

            result = run(
                run_cfg_dict=config,
                task_adapter=task_adapter,
                runtime=runtime_adapter,
                llm=llm,
                memory=None,
            )

            write_json(task_result_dir / "formal_run_result.json", result)

            summary_rows.append(
                build_summary_row(
                    task_index=idx,
                    raw_task=raw_task,
                    result=result,
                )
            )
            detailed_results.append(
                {
                    "task_index": idx,
                    "task_id": task_id,
                    "raw_task": raw_task,
                    "result": result,
                }
            )

            print("[OK] Formal run result:")
            print(json.dumps(result, ensure_ascii=False, indent=2))

        except Exception as e:
            err = repr(e)
            traceback.print_exc()

            fallback_result = {
                "status": "runner_failed",
                "mode": "formal",
                "run_dir": None,
                "task_id": task_id,
                "rounds": 0,
                "verifier_ok": None,
                "exec_status": None,
                "error_stage": "runner",
                "error": err,
            }

            write_json(task_result_dir / "formal_run_result.json", fallback_result)

            summary_rows.append(
                build_summary_row(
                    task_index=idx,
                    raw_task=raw_task,
                    result=fallback_result,
                    error=err,
                )
            )
            detailed_results.append(
                {
                    "task_index": idx,
                    "task_id": task_id,
                    "raw_task": raw_task,
                    "result": fallback_result,
                }
            )

            print(f"[FAIL] Task runner failed: {err}")

    summary_json = args.output_dir / "connectivity_summary.json"
    summary_csv = args.output_dir / "connectivity_summary.csv"
    detailed_json = args.output_dir / "connectivity_detailed_results.json"

    write_json(summary_json, summary_rows)
    write_csv(summary_csv, summary_rows)
    write_json(detailed_json, detailed_results)

    p("[FINAL] connectivity_summary.json", str(summary_json))
    p("[FINAL] connectivity_summary.csv", str(summary_csv))

    passed = sum(1 for row in summary_rows if row.get("status") == "ok")
    print("\n" + "=" * 100)
    print(f"Connectivity experiment finished: {passed}/{len(summary_rows)} tasks reached status=ok")
    print("=" * 100)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())