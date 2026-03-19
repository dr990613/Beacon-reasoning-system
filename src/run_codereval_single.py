# src/run_codereval_single.py
# -*- coding: utf-8 -*-

"""
Run a single CodeEval task through Beacon agent workflow and export
submission-style JSON:

{
  "question_id": "...",
  "answer": "..."
}

Current scope:
- read one task from CoderEval JSON
- build TaskObject + ProjectIndex directly from task record
- run AgentWorkflow only
- do NOT require local repo runtime / CodeEval execution environment
- write minimal submission JSON
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import yaml

from beacon_system.agents.workflow import AgentWorkflow
from beacon_system.llm.client import LLMClient
from beacon_system.llm.config import ModelConfig
from beacon_system.types import (
    AgentConfig,
    ProjectIndex,
    ReaderConfig,
    RunConfig,
    RuntimeConfig,
    TaskObject,
)


# ============================================================
# Basic helpers
# ============================================================


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_json(path: str, obj: Any) -> None:
    _ensure_dir(os.path.dirname(os.path.abspath(path)))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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


def _safe_text(text: Any) -> str:
    return str(text or "").strip()


# ============================================================
# Config builders
# ============================================================


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


def _build_run_config(cfg: Dict[str, Any], model_cfg: ModelConfig, output_dir: str) -> RunConfig:
    seed = _int_from_any(cfg.get("seed"), 42)
    return RunConfig(
        seed=seed,
        outputs_dir=output_dir,
        reader=_build_reader_config(cfg),
        model=model_cfg,
        agent=_build_agent_config(cfg),
        runtime=RuntimeConfig(
            work_dir="",
            run_command=(),
            env={},
            timeout_sec=None,
        ),
        adapter={},
        meta={},
    )


# ============================================================
# CodeEval task loading
# ============================================================


def _extract_records(data: Any) -> List[Dict[str, Any]]:
    """
    Support both:
    1) top-level list
    2) {"RECORDS": [...]}
    """
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        records = data.get("RECORDS")
        if isinstance(records, list):
            return [x for x in records if isinstance(x, dict)]

    raise ValueError("Unsupported CoderEval JSON format. Expected list or {'RECORDS': [...]}.")


def _find_task(records: List[Dict[str, Any]], task_id: str) -> Dict[str, Any]:
    for item in records:
        if str(item.get("_id") or "") == str(task_id):
            return item
    raise ValueError(f"task_id not found: {task_id}")


def _extract_spec(record: Dict[str, Any]) -> str:
    candidates = [
        record.get("docstring"),
        record.get("comment"),
        record.get("human_label"),
    ]
    for c in candidates:
        text = _safe_text(c)
        if text:
            return text
    return ""


def _extract_file_content(record: Dict[str, Any]) -> str:
    file_content = _safe_text(record.get("file_content"))
    if file_content:
        return file_content

    code = _safe_text(record.get("code"))
    if code:
        return code

    return ""


def _extract_target_file(record: Dict[str, Any]) -> str:
    path = _safe_text(record.get("file_path"))
    if path:
        return path.replace("\\", "/")
    return ""


def _extract_target_qualname(record: Dict[str, Any]) -> str:
    name = _safe_text(record.get("name"))
    if name:
        return name
    return ""


def build_task_from_codereval_record(record: Dict[str, Any]) -> TaskObject:
    task_id = _safe_text(record.get("_id"))
    target_file = _extract_target_file(record)
    target_qualname = _extract_target_qualname(record)

    return TaskObject(
        id=task_id,
        lang="python",
        level="function",
        target={
            "file": target_file,
            "qualname": target_qualname,
        },
        spec=_extract_spec(record),
        context={
            "docstring": _safe_text(record.get("docstring")),
            "human_label": _safe_text(record.get("human_label")),
            "project": _safe_text(record.get("project")),
            "package": _safe_text(record.get("package")),
            "file_path": target_file,
            "all_context": _safe_text(record.get("all_context")),
            "oracle_context": _safe_text(record.get("oracle_context")),
            "dependency": _safe_text(record.get("dependency")),
        },
        meta={
            "source": "codereval",
            "project": _safe_text(record.get("project")),
            "package": _safe_text(record.get("package")),
            "level": _safe_text(record.get("level")),
            "lineno": _safe_text(record.get("lineno")),
            "end_lineno": _safe_text(record.get("end_lineno")),
        },
    )


def build_project_index_from_codereval_record(record: Dict[str, Any]) -> ProjectIndex:
    target_file = _extract_target_file(record)
    target_qualname = _extract_target_qualname(record)
    file_content = _extract_file_content(record)

    return ProjectIndex(
        root=_safe_text(record.get("project")) or "codereval",
        entry_file=target_file,
        entry_qualname=target_qualname,
        files={target_file: file_content} if target_file else {},
        ast_index={},
        symbols={},
        callgraph={},
        meta={
            "source": "codereval",
            "project": _safe_text(record.get("project")),
            "package": _safe_text(record.get("package")),
            "task_id": _safe_text(record.get("_id")),
        },
    )


# ============================================================
# Main
# ============================================================


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one CodeEval task through Beacon workflow and export submission JSON."
    )
    parser.add_argument(
        "--task-id",
        required=True,
        help="CoderEval task _id.",
    )
    parser.add_argument(
        "--json-path",
        required=True,
        help="Path to CoderEval JSON file.",
    )
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/codereval_single",
        help="Directory to store generated JSON output.",
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
    args = build_arg_parser().parse_args()

    cfg = _load_yaml(args.config)
    model_cfg = ModelConfig.from_config_dict(cfg)
    model_cfg.validate()

    llm = LLMClient(cfg=model_cfg)
    run_config = _build_run_config(
        cfg=cfg,
        model_cfg=model_cfg,
        output_dir=str(args.output_dir),
    )

    data = _load_json(args.json_path)
    records = _extract_records(data)

    record = _find_task(records, args.task_id)
    task = build_task_from_codereval_record(record)
    project_index = build_project_index_from_codereval_record(record)

    workflow = AgentWorkflow(
        llm=llm,
        memory_store_path=str(args.memory_store_path),
        print_io=bool(args.print_io),
    )

    result = workflow.run(
        task=task,
        project_index=project_index,
        run_id="",
        run_config=run_config,
    )

    answer = ""
    if result.final_generation is not None and result.final_generation.primary is not None:
        answer = _safe_text(result.final_generation.primary.content)

    submission = {
        "question_id": str(task.id),
        "answer": answer,
    }

    output_path = os.path.join(
        str(args.output_dir),
        f"{task.id}.json",
    )
    _write_json(output_path, submission)

    print("=" * 80)
    print("Beacon CodeEval single-task generation finished")
    print(f"question_id : {task.id}")
    print(f"output_path : {output_path}")
    print(f"answer_len  : {len(answer)}")
    print(f"workflow_ok : {result.success}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())