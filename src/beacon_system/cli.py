# -*- coding: utf-8 -*-

"""
CLI entry for Beacon system.

Responsibilities:
- parse arguments
- load config
- initialize objects
- call pipeline
- print/save results

Non-goals:
- no business logic
- no Beacon reasoning
- no workflow details here
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .llm.config import load_llm_config, load_runtime_config, resolve_output_dir
from .llm.client import LLMClient
from .io import dump_json, ensure_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beacon-system",
        description="Beacon system command line entry.",
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config file.",
    )
    parser.add_argument(
        "--task-json",
        type=str,
        default=None,
        help="Path to a single task json file.",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=None,
        help="Path to local repository root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional override for output directory.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty print final result.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Save final pipeline result to output dir.",
    )
    return parser


def _read_json(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("task-json must contain a single task object (dict).")
    return data


def _resolve_output_dir(args: argparse.Namespace, config_path: str) -> str:
    if args.output_dir:
        return str(args.output_dir)
    return resolve_output_dir(config_path, default="outputs")


def _import_pipeline_components() -> Dict[str, Any]:
    """
    Lazy imports so CLI stays thin and avoids import-time coupling.
    Expected modules:
    - adapters/localrepo/task_adapter.py
    - logic/engine.py
    - pipeline.py
    """
    from .adapters.localrepo.task_adapter import LocalRepoTaskAdapter
    from .logic.engine import Engine
    from .pipeline import BeaconPipeline

    return {
        "TaskAdapter": LocalRepoTaskAdapter,
        "LogicEngine": Engine,
        "Pipeline": BeaconPipeline,
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    config_path = str(Path(args.config).resolve())
    runtime_config = load_runtime_config(config_path)
    llm_config = load_llm_config(config_path)

    llm_debug_dir = None
    artifacts_cfg = runtime_config.get("artifacts", {})
    if isinstance(artifacts_cfg, dict):
        llm_debug_dir = artifacts_cfg.get("llm_debug_dir")

    llm_client = LLMClient(
        llm_config,
        debug_dir=llm_debug_dir,
    )

    components = _import_pipeline_components()
    task_adapter = components["TaskAdapter"]()
    logic_engine = components["LogicEngine"]()
    pipeline = components["Pipeline"](
        task_adapter=task_adapter,
        logic_engine=logic_engine,
        llm_client=llm_client,
    )

    raw_task = None
    if args.task_json:
        raw_task = _read_json(args.task_json)

    result = pipeline.run(
        raw_task=raw_task,
        project_root=args.project_root,
        run_config=runtime_config,
    )

    result_dict = result.to_dict() if hasattr(result, "to_dict") else result

    if args.pretty:
        print(json.dumps(result_dict, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(result_dict, ensure_ascii=False))

    if args.save_result:
        output_dir = _resolve_output_dir(args, config_path)
        ensure_dir(output_dir)

        task_id = None
        if isinstance(result_dict, dict):
            task_id = result_dict.get("task_id")
            if task_id is None:
                task_id = result_dict.get("task", {}).get("task_id") if isinstance(result_dict.get("task"), dict) else None

        filename = f"{task_id or 'run_result'}.json"
        dump_json(Path(output_dir) / filename, result_dict)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())