# src/run_codereval_export_generations.py
# -*- coding: utf-8 -*-

"""
Export real Beacon-agent generations into minimal CoderEval-style JSON files.

Target format:
{
  "_id": "62e60f43d76274f8a4026e28",
  "generate_results": [
    "def ...",
    "def ..."
  ]
}

Design:
- REAL pipeline call only, no mock path
- verbose tracing for every stage
- debug mode: limit to 3 tasks
- full mode: process all tasks
"""

from __future__ import annotations

import os
import json
import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List


# =========================
# Real imports from your repo
# =========================
try:
    from beacon_system.pipeline import run_pipeline
except Exception as e:
    run_pipeline = None
    _RUN_PIPELINE_IMPORT_ERROR = e
else:
    _RUN_PIPELINE_IMPORT_ERROR = None


def load_config_stub(config_path: str) -> Dict[str, Any]:
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# =========================
# Helpers
# =========================
def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def short_text(text: Any, limit: int = 500) -> str:
    s = str(text)
    if len(s) <= limit:
        return s
    return s[:limit] + "\n...[TRUNCATED]..."


def print_banner(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_kv(key: str, value: Any) -> None:
    print(f"[TRACE] {key}: {value}")


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```") and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


def normalize_candidate(code: str) -> str:
    return strip_code_fence(code).strip()


def task_id_of(task: Dict[str, Any]) -> str:
    for k in ("_id", "task_id", "id", "question_id"):
        if k in task:
            return str(task[k])

    for k, v in task.items():
        if "id" in k.lower():
            return str(v)

    raise KeyError(f"Task has no recognizable id field. Keys: {list(task.keys())}")


def load_tasks_from_json(json_path: str) -> List[Dict[str, Any]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        print("[TRACE] detected top-level list task structure")
        return data

    if isinstance(data, dict):
        # CoderEval official structure
        if "RECORDS" in data and isinstance(data["RECORDS"], list):
            print("[TRACE] detected task list key: RECORDS")
            return data["RECORDS"]

        # Common wrappers
        for key in ("tasks", "data", "instances", "records", "Items", "items"):
            if key in data and isinstance(data[key], list):
                print(f"[TRACE] detected task list key: {key}")
                return data[key]

        # Fallback: first list[dict]
        for key, value in data.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                print(f"[TRACE] auto-detected task list key: {key}")
                return value

        # Single task dict fallback
        if "_id" in data or "task_id" in data or "id" in data:
            print("[TRACE] detected single task dict, wrapping into a list")
            return [data]

        raise ValueError(
            f"Unsupported dict json structure. Top-level keys: {list(data.keys())}"
        )

    raise ValueError(f"Unsupported task json structure: {type(data)}")


def extract_generate_results(pipeline_result: Any) -> List[str]:
    if pipeline_result is None:
        return []

    collected: List[str] = []

    if isinstance(pipeline_result, dict):
        # 1) top-level direct fields
        for key in ("generate_results", "candidates", "outputs"):
            value = pipeline_result.get(key)
            if isinstance(value, list):
                collected.extend(
                    normalize_candidate(str(x)) for x in value if x is not None
                )

        if pipeline_result.get("final_code"):
            collected.append(normalize_candidate(str(pipeline_result["final_code"])))

        # 2) rounds direct values
        rounds = pipeline_result.get("rounds")
        if isinstance(rounds, list):
            for r in rounds:
                if not isinstance(r, dict):
                    continue
                for key in ("generate_results", "candidates", "outputs"):
                    value = r.get(key)
                    if isinstance(value, list):
                        collected.extend(
                            normalize_candidate(str(x)) for x in value if x is not None
                        )
                if r.get("final_code"):
                    collected.append(normalize_candidate(str(r["final_code"])))

        # 3) artifacts direct values
        artifacts = pipeline_result.get("artifacts")
        if isinstance(artifacts, dict):
            # read all code_round*.py in key order
            code_keys = sorted(k for k in artifacts.keys() if k.startswith("code_round"))
            for key in code_keys:
                path = artifacts.get(key)
                if path and os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as f:
                            code = f.read().strip()
                        if code:
                            collected.append(normalize_candidate(code))
                    except Exception as e:
                        print(f"[TRACE] failed to read artifact {key}: {e}")

            # fallback: read generation jsons only if no code collected
            if not collected:
                gen_keys = sorted(k for k in artifacts.keys() if k.startswith("generation_round"))
                for key in gen_keys:
                    path = artifacts.get(key)
                    if path and os.path.exists(path):
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                obj = json.load(f)
                            collected.append(json.dumps(obj, ensure_ascii=False, indent=2))
                        except Exception as e:
                            print(f"[TRACE] failed to read artifact {key}: {e}")

    elif isinstance(pipeline_result, list):
        collected.extend(
            normalize_candidate(str(x)) for x in pipeline_result if x is not None
        )

    # deduplicate while preserving order
    deduped: List[str] = []
    seen = set()
    for item in collected:
        if item and item not in seen:
            deduped.append(item)
            seen.add(item)

    return deduped


def trace_pipeline_result(pipeline_result: Any) -> None:
    print_banner("PIPELINE RAW RESULT SUMMARY")

    artifacts = pipeline_result.get("artifacts")
    if isinstance(artifacts, dict):
        for k, v in artifacts.items():
            print_kv(f"artifact.{k}", v)

    print_kv("pipeline_result_type", type(pipeline_result).__name__)

    if isinstance(pipeline_result, dict):
        print_kv("pipeline_result_keys", list(pipeline_result.keys()))

        for key in (
                "task_id",
                "status",
                "mode",
                "run_dir",
                "rounds",
                "artifacts",
                "verifier_ok",
                "exec_status",
                "error_stage",
                "error",
                "final_code",
                "generate_results",
                "candidates",
                "outputs",
                "beacon_ir",
                "constraints",
                "verification",
                "verifier_result",
                "memory",
        ):
            if key in pipeline_result:
                value = pipeline_result[key]
                if isinstance(value, list):
                    print_kv(f"{key}_len", len(value))
                    if value:
                        print(f"[TRACE] {key}[0] preview:\n{short_text(value[0], 600)}")
                else:
                    print(f"[TRACE] {key} preview:\n{short_text(value, 800)}")
    else:
        print(short_text(pipeline_result, 1000))


def save_minimal_json(task_id: str, generate_results: List[str], output_dir: str) -> str:
    payload = {
        "_id": task_id,
        "generate_results": generate_results,
    }
    out_path = os.path.join(output_dir, f"{task_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def append_jsonl(task_id: str, generate_results: List[str], jsonl_path: str) -> None:
    payload = {
        "_id": task_id,
        "generate_results": generate_results,
    }
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def build_task_payload_for_trace(task: Dict[str, Any]) -> Dict[str, Any]:
    trace_payload = {
        "_id": task.get("_id"),
        "name": task.get("name"),
        "project": task.get("project"),
        "file_path": task.get("file_path"),
        "lineno": task.get("lineno"),
        "end_lineno": task.get("end_lineno"),
        "test_name": task.get("test_name"),
    }
    return trace_payload


# =========================
# Main real call
# =========================
def run_one_task_real(task: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    This function MUST call the real Beacon pipeline.
    Replace only the invocation signature here if your real pipeline signature differs.
    """
    if run_pipeline is None:
        raise RuntimeError(
            f"run_pipeline import failed: {_RUN_PIPELINE_IMPORT_ERROR}"
        )

    tid = task_id_of(task)

    print_banner(f"RUN TASK: {tid}")
    print_kv("task_id", tid)
    print_kv("task_keys", list(task.keys()))
    print_kv("task_summary", build_task_payload_for_trace(task))

    for k in ("docstring", "name", "file_path", "project", "package", "all_context", "oracle_context"):
        if k in task:
            print(f"[TRACE] task.{k} preview:\n{short_text(task[k], 800)}")

    print_banner("PIPELINE START")

    # ===== Real pipeline call =====
    # If your real signature differs, change ONLY this line.
    result = run_pipeline(task=task, config=config)

    print_banner("PIPELINE END")
    trace_pipeline_result(result)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Export real Beacon generations for CoderEval.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--json-path", required=True, help="Path to CoderEval json")
    parser.add_argument("--output-dir", default="outputs/generations")
    parser.add_argument("--jsonl-path", default="outputs/generations/all_results.jsonl")
    parser.add_argument(
        "--debug-three-tasks",
        action="store_true",
        help="If set, only run the first 3 tasks",
    )
    parser.add_argument(
        "--disable-debug-limit",
        action="store_true",
        help="If set, ignore the 3-task limit and run all tasks",
    )
    parser.add_argument(
        "--task-id",
        default=None,
        help="Run only one specific task id",
    )
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    ensure_dir(str(Path(args.jsonl_path).parent))

    print_banner("EXPORT GENERATIONS CONFIG")
    print_kv("config", args.config)
    print_kv("json_path", args.json_path)
    print_kv("output_dir", args.output_dir)
    print_kv("jsonl_path", args.jsonl_path)
    print_kv("debug_three_tasks", args.debug_three_tasks)
    print_kv("disable_debug_limit", args.disable_debug_limit)
    print_kv("task_id", args.task_id)

    config = load_config_stub(args.config)
    tasks = load_tasks_from_json(args.json_path)

    print_kv("loaded_tasks", len(tasks))
    if tasks:
        print_kv("first_task_keys", list(tasks[0].keys()))
        print_kv("first_task_id", task_id_of(tasks[0]))

    if args.task_id:
        tasks = [t for t in tasks if task_id_of(t) == args.task_id]
        print_kv("filtered_tasks", len(tasks))

    use_limit_3 = args.debug_three_tasks and not args.disable_debug_limit
    if use_limit_3:
        tasks = tasks[:3]
        print_kv("effective_task_limit", 3)
    else:
        print_kv("effective_task_limit", "ALL")

    success = 0
    failed = 0

    for idx, task in enumerate(tasks, start=1):
        tid = task_id_of(task)
        print_banner(f"[{idx}/{len(tasks)}] PROCESS TASK {tid}")

        try:
            result = run_one_task_real(task=task, config=config)
            generate_results = extract_generate_results(result)

            print_banner("EXTRACTED GENERATIONS")
            print_kv("task_id", tid)
            print_kv("num_generate_results", len(generate_results))

            for i, code in enumerate(generate_results):
                print(f"\n[TRACE] candidate #{i}\n{'-' * 80}")
                print(short_text(code, 2000))

            out_path = save_minimal_json(
                task_id=tid,
                generate_results=generate_results,
                output_dir=args.output_dir,
            )
            append_jsonl(
                task_id=tid,
                generate_results=generate_results,
                jsonl_path=args.jsonl_path,
            )

            print_kv("saved_json", out_path)
            print_kv("saved_jsonl_append", args.jsonl_path)
            success += 1

        except Exception as e:
            failed += 1
            print_banner(f"[ERROR] TASK FAILED: {tid}")
            print_kv("error_type", type(e).__name__)
            print_kv("error_message", str(e))
            print("[TRACE] traceback:")
            print(traceback.format_exc())

    print_banner("EXPORT SUMMARY")
    print_kv("success", success)
    print_kv("failed", failed)
    print_kv("total", len(tasks))


if __name__ == "__main__":
    main()