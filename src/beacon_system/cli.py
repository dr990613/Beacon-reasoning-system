# src/beacon_system/cli.py
# -*- coding: utf-8 -*-

"""
CLI entry for Beacon reasoning system.

Scope:
- load yaml config
- build model config + llm client
- build adapters
- build run config
- run pipeline once

Non-goals:
- no benchmark registry here
- no business logic here
- no prompt logic here
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import yaml

from .adapters.localrepo.runtime import LocalRepoRuntimeAdapter
from .adapters.localrepo.task_adapter import LocalRepoTaskAdapter
from .llm.client import LLMClient
from .llm.config import ModelConfig
from .pipeline import run_pipeline
from .types import AgentConfig, ReaderConfig, RunConfig, RuntimeConfig


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def _bool_from_any(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _int_from_any(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except Exception:
        return default


def _optional_int_from_any(value: Any) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _build_reader_config(cfg: Dict[str, Any]) -> ReaderConfig:
    reader = dict(cfg.get("reader") or {})
    return ReaderConfig(
        enable_global=_bool_from_any(reader.get("enable_global"), True),
        validation_filter=_bool_from_any(reader.get("validation_filter"), True),
        max_local_nodes=_optional_int_from_any(reader.get("max_local_nodes")),
        max_global_inline=_optional_int_from_any(reader.get("max_global_inline")),
    )


def _build_agent_config(cfg: Dict[str, Any]) -> AgentConfig:
    agent = dict(cfg.get("agent") or {})
    return AgentConfig(
        max_rounds=_int_from_any(agent.get("max_rounds"), 2),
        max_thoughts=_int_from_any(agent.get("max_thoughts"), 3),
        keep_top_k=_int_from_any(agent.get("keep_top_k"), 1),
        use_memory=_bool_from_any(agent.get("use_memory"), True),
        use_verifier=_bool_from_any(agent.get("use_verifier"), True),
        require_logic_acceptance=_bool_from_any(agent.get("require_logic_acceptance"), True),
        require_beacon_usage_check=_bool_from_any(agent.get("require_beacon_usage_check"), True),
    )


def _build_runtime_config(cfg: Dict[str, Any]) -> RuntimeConfig:
    runtime = dict(cfg.get("runtime") or {})
    run_command = runtime.get("run_command") or runtime.get("run_cmd") or "pytest -q"

    if isinstance(run_command, (list, tuple)):
        run_command_tuple = tuple(str(x) for x in run_command)
    else:
        run_command_tuple = (str(run_command),)

    return RuntimeConfig(
        work_dir=str(runtime.get("work_dir") or ""),
        run_command=run_command_tuple,
        env=dict(runtime.get("env") or {}),
        timeout_sec=_optional_int_from_any(runtime.get("timeout_sec")),
    )


def _build_run_config(cfg: Dict[str, Any], model_cfg: ModelConfig) -> RunConfig:
    outputs_dir = str(cfg.get("outputs_dir") or "outputs/runs")
    seed = _int_from_any(cfg.get("seed"), 42)

    return RunConfig(
        seed=seed,
        outputs_dir=outputs_dir,
        reader=_build_reader_config(cfg),
        model=model_cfg,
        agent=_build_agent_config(cfg),
        runtime=_build_runtime_config(cfg),
        adapter=dict(cfg.get("adapter") or {}),
        meta=dict(cfg.get("meta") or {}),
    )


def _runtime_cmd_as_str(runtime_cfg: RuntimeConfig) -> str:
    if not runtime_cfg.run_command:
        return "pytest -q"
    if len(runtime_cfg.run_command) == 1:
        return str(runtime_cfg.run_command[0])
    return " ".join(runtime_cfg.run_command)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Beacon reasoning system CLI"
    )

    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--repo-root",
        required=True,
        help="Project/repository root for localrepo adapter.",
    )
    parser.add_argument(
        "--target-file",
        required=True,
        help="Target file relative to repo root.",
    )
    parser.add_argument(
        "--target-qualname",
        required=True,
        help="Target qualname inside target file.",
    )
    parser.add_argument(
        "--spec",
        default="",
        help="Task spec / instruction text.",
    )
    parser.add_argument(
        "--task-id",
        default="",
        help="Optional explicit task id.",
    )
    parser.add_argument(
        "--lang",
        default="",
        help="Optional task language override.",
    )
    parser.add_argument(
        "--level",
        default="",
        help="Optional task level override, e.g. function/file/project.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional explicit run id.",
    )
    parser.add_argument(
        "--memory-store-path",
        default="outputs/memory/experience.jsonl",
        help="Path to local JSONL experience memory store.",
    )
    parser.add_argument(
        "--print-io",
        action="store_true",
        help="Enable lightweight debug prints.",
    )

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = _load_yaml(args.config)

    model_cfg = ModelConfig.from_config_dict(cfg)
    model_cfg.validate()

    llm = LLMClient(cfg=model_cfg)

    run_config = _build_run_config(cfg, model_cfg)

    task_meta: Dict[str, Any] = {}
    if args.task_id:
        task_meta["id"] = str(args.task_id)
    if args.lang:
        task_meta["lang"] = str(args.lang)
    if args.level:
        task_meta["level"] = str(args.level)

    task_adapter = LocalRepoTaskAdapter(
        repo_root=str(args.repo_root),
        target_file=str(args.target_file),
        target_qualname=str(args.target_qualname),
        spec=str(args.spec or ""),
        context={},
        meta=task_meta,
    )

    runtime_adapter = LocalRepoRuntimeAdapter(
        repo_root=str(args.repo_root),
        run_cmd=_runtime_cmd_as_str(run_config.runtime),
        work_dir=run_config.runtime.work_dir or None,
        timeout_sec=run_config.runtime.timeout_sec,
        print_io=bool(args.print_io),
    )

    result = run_pipeline(
        llm=llm,
        task_adapter=task_adapter,
        runtime_adapter=runtime_adapter,
        config=run_config,
        run_id=(str(args.run_id).strip() or None),
        memory_store_path=str(args.memory_store_path),
        print_io=bool(args.print_io),
    )

    run_id = str((result.meta or {}).get("run_id") or "")
    run_dir = str((result.meta or {}).get("run_dir") or "")
    runtime_status = ""
    if result.final_exec is not None:
        runtime_status = f"{result.final_exec.status} (rc={result.final_exec.return_code})"
    else:
        runtime_status = "not-run"

    print("=" * 80)
    print("Beacon pipeline finished")
    print(f"success        : {result.success}")
    print(f"task_id        : {result.task.id}")
    print(f"target         : {result.task.target.get('file')}::{result.task.target.get('qualname')}")
    print(f"run_id         : {run_id}")
    print(f"run_dir        : {run_dir}")
    print(f"rounds         : {len(result.rounds or ())}")
    print(f"runtime_status : {runtime_status}")
    print("=" * 80)

    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())