# -*- coding: utf-8 -*-

"""
Full pass@10 generation runner for CoderEval.

Responsibilities:
- read all tasks from Python/Java benchmark json files
- do NOT depend on local workspace/project root
- run agent workflow 10 times per task
- save jsonl rows with:
    id, answer_1, answer_2, ..., answer_10
- save per-pass artifacts
- save three merged files:
    1. python_predictions_pass10.jsonl
    2. java_predictions_pass10.jsonl
    3. all_predictions_pass10.jsonl
- continue on single-task failure and record errors
"""

from __future__ import annotations

import argparse
import copy
import json
import traceback
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from beacon_system.adapters.localrepo.task_adapter import LocalRepoTaskAdapter
from beacon_system.agents.workflow import AgentWorkflow
from beacon_system.io import (
    save_generation_artifacts,
    save_logic_artifacts,
    save_run_trace,
    save_verification_artifacts,
)
from beacon_system.llm.client import LLMClient
from beacon_system.llm.config import load_llm_config, load_runtime_config
from beacon_system.types import ProjectIndex
import beacon_system.logic.engine as logic_engine


# ============================================================
# basic io
# ============================================================

def _read_json(path: str | Path) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_jsonl_row(output_path: Path, row: Dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return output_path


def _append_jsonl(output_path: Path, row: Dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _append_text(output_path: Path, text: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        f.write(text)


# ============================================================
# benchmark parsing
# ============================================================

def _extract_records(data: Any) -> List[Dict[str, Any]]:
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict) and isinstance(data.get("RECORDS"), list):
        records = data["RECORDS"]
    else:
        raise ValueError("Unsupported benchmark json format. Expected list or {'RECORDS': [...]}.")

    return [x for x in records if isinstance(x, dict)]


def _pick_task_id(task: Dict[str, Any]) -> str:
    value = task.get("task_id") or task.get("_id")
    if value is None:
        raise ValueError("Task has no 'task_id' or '_id'.")
    return str(value)


def _load_all_tasks(
    *,
    python_json: str | Path,
    java_json: str | Path,
) -> List[Tuple[str, Dict[str, Any]]]:
    py_data = _read_json(python_json)
    java_data = _read_json(java_json)

    py_records = _extract_records(py_data)
    java_records = _extract_records(java_data)

    selected: List[Tuple[str, Dict[str, Any]]] = []
    selected.extend([("python", x) for x in py_records])
    selected.extend([("java", x) for x in java_records])
    return selected


# ============================================================
# minimal project index
# ============================================================

def _project_index_field_names() -> set[str]:
    if is_dataclass(ProjectIndex):
        return {f.name for f in fields(ProjectIndex)}
    return set()


def _make_minimal_project_index(task: Any, raw_task: Dict[str, Any]) -> ProjectIndex:
    """
    Build a minimal JSON-only ProjectIndex.
    No local repo required.
    """
    target_file = (
        getattr(task, "target_file", None)
        or getattr(task, "file_path", None)
        or raw_task.get("target_file")
        or raw_task.get("file_path")
    )
    file_content = (
        getattr(task, "file_content", None)
        or raw_task.get("file_content")
        or ""
    )
    code = (
        getattr(task, "code", None)
        or raw_task.get("code")
        or ""
    )
    task_id = getattr(task, "task_id", None) or raw_task.get("task_id") or raw_task.get("_id")
    lang = getattr(task, "lang", None) or raw_task.get("lang") or raw_task.get("language")
    qualname = getattr(task, "qualname", None)
    target_name = (
        getattr(task, "target_name", None)
        or getattr(task, "target_function", None)
        or getattr(task, "name", None)
        or raw_task.get("name")
    )

    relevant_blocks: List[Dict[str, Any]] = []
    if isinstance(file_content, str) and file_content.strip():
        relevant_blocks.append({
            "kind": "target_file",
            "file_path": target_file,
            "content": file_content,
        })
    if isinstance(code, str) and code.strip():
        relevant_blocks.append({
            "kind": "target_code",
            "file_path": target_file,
            "content": code,
        })

    candidate_kwargs: Dict[str, Any] = {
        "project_root": "<json-only>",
        "project_name": raw_task.get("project"),
        "language": lang,
        "files": [target_file] if isinstance(target_file, str) and target_file.strip() else [],
        "file_texts": (
            {str(target_file): file_content}
            if isinstance(target_file, str) and target_file.strip() and isinstance(file_content, str)
            else {}
        ),
        "relevant_blocks": relevant_blocks,
        "target_file": target_file,
        "qualname": qualname or target_name,
        "task_id": str(task_id) if task_id is not None else None,
    }

    allowed = _project_index_field_names()
    if allowed:
        candidate_kwargs = {k: v for k, v in candidate_kwargs.items() if k in allowed}

    return ProjectIndex(**candidate_kwargs)  # type: ignore[arg-type]


# ============================================================
# workflow artifact save
# ============================================================

def _extract_round_dict(round_result: Any) -> Dict[str, Any]:
    if round_result is None:
        return {}
    if hasattr(round_result, "to_dict"):
        try:
            data = round_result.to_dict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    if hasattr(round_result, "__dict__"):
        return dict(vars(round_result))
    if isinstance(round_result, dict):
        return round_result
    return {}


def _save_workflow_artifacts(
    *,
    output_dir: str | Path,
    task_id: str,
    task: Any,
    project_index: Any,
    workflow_result: Any,
) -> Dict[str, Any]:
    artifact_paths: Dict[str, Any] = {
        "logic": {},
        "main_round": {},
        "revise_round": {},
        "run_trace": None,
    }

    logic_result = getattr(workflow_result, "logic_result", None)
    if logic_result is not None:
        artifact_paths["logic"] = save_logic_artifacts(
            output_dir=output_dir,
            task_id=task_id,
            logic_result=logic_result,
        )

    main_round = _extract_round_dict(getattr(workflow_result, "main_round", None))
    if main_round:
        generation = main_round.get("generation")
        verification = main_round.get("verification")

        if generation is not None:
            artifact_paths["main_round"]["generation"] = save_generation_artifacts(
                output_dir=output_dir,
                task_id=task_id,
                generation_result=generation,
                round_name="main_round",
            )

        if verification is not None:
            artifact_paths["main_round"]["verification"] = save_verification_artifacts(
                output_dir=output_dir,
                task_id=task_id,
                verification_result=verification,
                round_name="main_round",
            )

    revise_round = _extract_round_dict(getattr(workflow_result, "revise_round", None))
    if revise_round:
        generation = revise_round.get("generation")
        verification = revise_round.get("verification")

        if generation is not None:
            artifact_paths["revise_round"]["generation"] = save_generation_artifacts(
                output_dir=output_dir,
                task_id=task_id,
                generation_result=generation,
                round_name="revise_round",
            )

        if verification is not None:
            artifact_paths["revise_round"]["verification"] = save_verification_artifacts(
                output_dir=output_dir,
                task_id=task_id,
                verification_result=verification,
                round_name="revise_round",
            )

    run_trace_payload = {
        "task_id": task_id,
        "task": task.to_dict() if hasattr(task, "to_dict") else vars(task),
        "project_index": project_index.to_dict() if hasattr(project_index, "to_dict") else vars(project_index),
        "workflow_result": workflow_result.to_dict() if hasattr(workflow_result, "to_dict") else vars(workflow_result),
    }

    artifact_paths["run_trace"] = save_run_trace(
        output_dir=output_dir,
        task_id=task_id,
        run_trace=run_trace_payload,
    )

    return artifact_paths


# ============================================================
# runner
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Full pass@10 generation runner for CoderEval.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config.")
    parser.add_argument("--python-json", type=str, required=True, help="Path to CoderEval4Python.json")
    parser.add_argument("--java-json", type=str, required=True, help="Path to CoderEval4Java.json")
    parser.add_argument("--pass-k", type=int, default=10, help="How many generations to run per task.")
    parser.add_argument("--output-dir", type=str, default="outputs/pass10_full", help="Output directory.")
    parser.add_argument("--pretty", action="store_true", help="Pretty print final summary.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    config_path = str(Path(args.config).resolve())
    python_json = str(Path(args.python_json).resolve())
    java_json = str(Path(args.java_json).resolve())
    output_dir = _ensure_dir(args.output_dir)

    runtime_config = load_runtime_config(config_path)
    if not isinstance(runtime_config, dict):
        runtime_config = {}

    artifacts_cfg = runtime_config.get("artifacts")
    if not isinstance(artifacts_cfg, dict):
        artifacts_cfg = {}
        runtime_config["artifacts"] = artifacts_cfg

    llm_config = load_llm_config(config_path)
    if not llm_config.api_key or not str(llm_config.api_key).strip():
        raise ValueError("LLM api_key is empty. Please set llm.api_key in config yaml.")

    llm_debug_dir = artifacts_cfg.get("llm_debug_dir")

    print("[LLM CONFIG]")
    print(f"base_url   : {llm_config.base_url}")
    print(f"model_name : {llm_config.model_name}")
    print(f"api_key set: {bool(str(llm_config.api_key).strip())}")
    print()

    llm_client = LLMClient(llm_config, debug_dir=llm_debug_dir)
    task_adapter = LocalRepoTaskAdapter()
    workflow = AgentWorkflow(
        logic_engine=logic_engine,
        llm_client=llm_client,
        allow_one_step_revise=True,
    )

    selected_tasks = _load_all_tasks(
        python_json=python_json,
        java_json=java_json,
    )

    python_merged_path = output_dir / "python_predictions_pass10.jsonl"
    java_merged_path = output_dir / "java_predictions_pass10.jsonl"
    all_merged_path = output_dir / "all_predictions_pass10.jsonl"
    error_jsonl_path = output_dir / "errors.jsonl"
    progress_log_path = output_dir / "progress.log"

    for path in [python_merged_path, java_merged_path, all_merged_path, error_jsonl_path, progress_log_path]:
        if path.exists():
            path.unlink()

    all_summaries: List[Dict[str, Any]] = []
    success_count = 0
    fail_count = 0

    total_tasks = len(selected_tasks)

    for task_idx, (lang_tag, raw_task) in enumerate(selected_tasks, start=1):
        task_id = _pick_task_id(raw_task)

        header = (
            "=" * 100 + "\n"
            f"[{task_idx}/{total_tasks}] full pass-k running\n"
            f"lang    : {lang_tag}\n"
            f"task_id : {task_id}\n"
            f"pass_k  : {args.pass_k}\n"
            + "=" * 100 + "\n"
        )
        print(header, end="")
        _append_text(progress_log_path, header)

        try:
            task = task_adapter.load_task(raw_task)
            project_index = _make_minimal_project_index(task, raw_task)

            row: Dict[str, Any] = {"id": task_id}
            pass_records: List[Dict[str, Any]] = []

            for pass_idx in range(1, args.pass_k + 1):
                pass_output_dir = output_dir / lang_tag / task_id / f"pass_{pass_idx}"

                pass_runtime_config = copy.deepcopy(runtime_config)
                pass_artifacts_cfg = pass_runtime_config.get("artifacts")
                if not isinstance(pass_artifacts_cfg, dict):
                    pass_artifacts_cfg = {}
                    pass_runtime_config["artifacts"] = pass_artifacts_cfg
                pass_artifacts_cfg["output_dir"] = str(pass_output_dir)

                msg = f"[TASK {task_idx}/{total_tasks}] [PASS {pass_idx}/{args.pass_k}] generating...\n"
                print(msg, end="")
                _append_text(progress_log_path, msg)

                workflow_result = workflow.run(
                    task=task,
                    project_index=project_index,
                    run_config=pass_runtime_config,
                )

                final_code = getattr(workflow_result, "final_code", "") or ""
                row[f"answer_{pass_idx}"] = final_code

                artifact_paths = _save_workflow_artifacts(
                    output_dir=pass_output_dir,
                    task_id=task_id,
                    task=task,
                    project_index=project_index,
                    workflow_result=workflow_result,
                )

                pass_records.append({
                    "pass_index": pass_idx,
                    "accepted": getattr(workflow_result, "accepted", None),
                    "total_rounds": getattr(workflow_result, "total_rounds", None),
                    "stopped_reason": getattr(workflow_result, "stopped_reason", None),
                    "run_trace": artifact_paths.get("run_trace"),
                    "artifact_root": str(pass_output_dir / task_id),
                })

            single_jsonl_path = output_dir / lang_tag / f"{task_id}_pass{args.pass_k}.jsonl"
            _write_jsonl_row(single_jsonl_path, row)

            if lang_tag == "python":
                _append_jsonl(python_merged_path, row)
            elif lang_tag == "java":
                _append_jsonl(java_merged_path, row)

            _append_jsonl(all_merged_path, row)

            summary = {
                "lang": lang_tag,
                "task_id": task_id,
                "pass_k": args.pass_k,
                "single_jsonl": str(single_jsonl_path),
                "passes": pass_records,
            }
            all_summaries.append(summary)

            done_msg = (
                "[TASK RECORD]\n"
                f"single jsonl : {single_jsonl_path}\n"
                f"python total : {python_merged_path}\n"
                f"java total   : {java_merged_path}\n"
                f"all total    : {all_merged_path}\n"
                f"artifact root: {output_dir / lang_tag / task_id}\n\n"
            )
            print(done_msg, end="")
            _append_text(progress_log_path, done_msg)
            success_count += 1

        except Exception as exc:
            fail_count += 1
            error_row = {
                "lang": lang_tag,
                "task_id": task_id,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            _append_jsonl(error_jsonl_path, error_row)

            err_msg = (
                "[TASK ERROR]\n"
                f"lang      : {lang_tag}\n"
                f"task_id   : {task_id}\n"
                f"error     : {exc}\n"
                f"errors log: {error_jsonl_path}\n\n"
            )
            print(err_msg, end="")
            _append_text(progress_log_path, err_msg)
            continue

    final_summary = {
        "pass_k": args.pass_k,
        "total_tasks": total_tasks,
        "success_count": success_count,
        "fail_count": fail_count,
        "python_predictions_jsonl": str(python_merged_path),
        "java_predictions_jsonl": str(java_merged_path),
        "all_predictions_jsonl": str(all_merged_path),
        "error_jsonl": str(error_jsonl_path),
        "progress_log": str(progress_log_path),
        "runs": all_summaries,
    }

    print("#" * 100)
    print("[FULL PASS-K SUMMARY]")
    if args.pretty:
        print(json.dumps(final_summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(json.dumps(final_summary, ensure_ascii=False))
    print("#" * 100)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("\n[OUTPUT FILES]")
    print(f"python total : {python_merged_path}")
    print(f"java total   : {java_merged_path}")
    print(f"all total    : {all_merged_path}")
    print(f"errors       : {error_jsonl_path}")
    print(f"progress log : {progress_log_path}")
    print(f"summary json : {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())