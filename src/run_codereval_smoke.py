# -*- coding: utf-8 -*-

"""
Single-task smoke runner for CoderEval.

Responsibilities:
- load one task from benchmark json
- run BeaconPipeline once
- write jsonl output with only: id, answer
- print saved run record locations

Usage example:
python .\src\run_codereval_smoke.py ^
  --config .\configs\default.yaml ^
  --benchmark-json .\benchmarks\CoderEval\CoderEval4Python.json ^
  --project-root .\benchmarks\CoderEval\workspace\neo4j-python-driver ^
  --task-id 62e60f43d76274f8a4026e28 ^
  --output-dir .\outputs\smoke ^
  --pretty
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from beacon_system.adapters.localrepo.task_adapter import LocalRepoTaskAdapter
from beacon_system.llm.client import LLMClient
from beacon_system.llm.config import load_llm_config, load_runtime_config
from beacon_system.pipeline import BeaconPipeline
from beacon_system.logic.engine import Engine


def _read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_records(data: Any) -> List[Dict[str, Any]]:
    """
    Support common benchmark formats:
    1. [ {...}, {...} ]
    2. { "RECORDS": [ {...}, {...} ] }
    """
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("RECORDS"), list):
        records = data["RECORDS"]
    else:
        raise ValueError(
            "Unsupported benchmark json format. Expected a list or a dict with key 'RECORDS'."
        )

    clean_records: List[Dict[str, Any]] = []
    for item in records:
        if isinstance(item, dict):
            clean_records.append(item)
    return clean_records


def _pick_task(records: List[Dict[str, Any]], task_id: Optional[str]) -> Dict[str, Any]:
    if not records:
        raise ValueError("No task records found in benchmark json.")

    if task_id is None:
        return records[0]

    for record in records:
        rid = record.get("task_id") or record.get("_id")
        if rid is not None and str(rid) == str(task_id):
            return record

    raise ValueError(f"Task id not found in benchmark json: {task_id}")


def _pick_task_id(task: Dict[str, Any]) -> str:
    value = task.get("task_id") or task.get("_id")
    if value is None:
        raise ValueError("Selected task has no 'task_id' or '_id'.")
    return str(value)


def _ensure_output_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_jsonl(output_path: Path, task_id: str, answer: str) -> Path:
    line = {
        "id": task_id,
        "answer": answer,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one CoderEval task through BeaconPipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    parser.add_argument("--benchmark-json", type=str, required=True, help="Path to CoderEval json.")
    parser.add_argument("--project-root", type=str, required=True, help="Path to project workspace root.")
    parser.add_argument("--task-id", type=str, default=None, help="Optional task id. If omitted, use first record.")
    parser.add_argument("--output-dir", type=str, default="outputs/smoke", help="Directory for smoke outputs.")
    parser.add_argument("--pretty", action="store_true", help="Pretty print final summary.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config_path = str(Path(args.config).resolve())
    benchmark_json = str(Path(args.benchmark_json).resolve())
    project_root = str(Path(args.project_root).resolve())
    output_dir = _ensure_output_dir(args.output_dir)

    benchmark_data = _read_json(benchmark_json)
    records = _extract_records(benchmark_data)
    raw_task = _pick_task(records, args.task_id)
    selected_task_id = _pick_task_id(raw_task)

    runtime_config = load_runtime_config(config_path)

    # Override artifacts output dir for this smoke run
    if not isinstance(runtime_config, dict):
        runtime_config = {}
    artifacts_cfg = runtime_config.get("artifacts")
    if not isinstance(artifacts_cfg, dict):
        artifacts_cfg = {}
        runtime_config["artifacts"] = artifacts_cfg
    artifacts_cfg["output_dir"] = str(output_dir)

    llm_config = load_llm_config(config_path)
    llm_debug_dir = artifacts_cfg.get("llm_debug_dir")

    llm_client = LLMClient(llm_config, debug_dir=llm_debug_dir)
    task_adapter = LocalRepoTaskAdapter()
    logic_engine = Engine()

    pipeline = BeaconPipeline(
        task_adapter=task_adapter,
        logic_engine=logic_engine,
        llm_client=llm_client,
    )

    result = pipeline.run(
        raw_task=raw_task,
        project_root=project_root,
        run_config=runtime_config,
    )

    result_dict = result.to_dict() if hasattr(result, "to_dict") else result
    workflow_result = result_dict.get("workflow_result", {}) if isinstance(result_dict, dict) else {}

    final_code = ""
    if isinstance(workflow_result, dict):
        final_code = str(workflow_result.get("final_code", ""))

    prediction_path = _write_jsonl(
        output_dir / f"{selected_task_id}.jsonl",
        task_id=selected_task_id,
        answer=final_code,
    )

    artifact_paths = result_dict.get("artifact_paths", {}) if isinstance(result_dict, dict) else {}
    run_trace_path = artifact_paths.get("run_trace")
    logic_paths = artifact_paths.get("logic", {})
    main_round_paths = artifact_paths.get("main_round", {})
    revise_round_paths = artifact_paths.get("revise_round", {})

    summary = {
        "task_id": selected_task_id,
        "accepted": workflow_result.get("accepted") if isinstance(workflow_result, dict) else None,
        "total_rounds": workflow_result.get("total_rounds") if isinstance(workflow_result, dict) else None,
        "stopped_reason": workflow_result.get("stopped_reason") if isinstance(workflow_result, dict) else None,
        "prediction_jsonl": str(prediction_path),
        "run_trace": run_trace_path,
        "logic_artifacts": logic_paths,
        "main_round_artifacts": main_round_paths,
        "revise_round_artifacts": revise_round_paths,
    }

    if args.pretty:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(summary, ensure_ascii=False))

    print("\n[SMOKE RUN RECORD]")
    print(f"prediction jsonl : {prediction_path}")
    print(f"run trace       : {run_trace_path}")
    print(f"artifact root    : {output_dir / selected_task_id}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())