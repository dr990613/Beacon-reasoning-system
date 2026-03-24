# run_codereval_baseline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config_baseline import load_config
from generator_baseline import BaselineGenerator
from pipeline_baseline import BaselinePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run baseline code generation on CoderEval tasks."
    )
    parser.add_argument(
        "--input-json",
        type=str,
        default=None,
        help="Single benchmark json file. Legacy mode. Overrides config if provided.",
    )
    parser.add_argument(
        "--python-json",
        type=str,
        default=None,
        help="Path to Python benchmark json file.",
    )
    parser.add_argument(
        "--java-json",
        type=str,
        default=None,
        help="Path to Java benchmark json file.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Path to output json file. Overrides config if provided.",
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Run only one task by id (_id/question_id/task_id/id). Only used in single-file mode.",
    )
    parser.add_argument(
        "--python-task-id",
        type=str,
        default=None,
        help="Run only one Python task by id.",
    )
    parser.add_argument(
        "--java-task-id",
        type=str,
        default=None,
        help="Run only one Java task by id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Legacy single-file mode: run only the first N tasks after filtering.",
    )
    parser.add_argument(
        "--take-n-each",
        type=int,
        default=1,
        help="How many tasks to take from Python and Java each. Default: 1.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Legacy single-file mode start index.",
    )
    parser.add_argument(
        "--python-start-index",
        type=int,
        default=0,
        help="Python start index in dual-language mode.",
    )
    parser.add_argument(
        "--java-start-index",
        type=int,
        default=0,
        help="Java start index in dual-language mode.",
    )
    parser.add_argument(
        "--num-passes",
        type=int,
        default=None,
        help="How many times to run each task. Overrides config if provided.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file instead of merging/updating existing results.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print output json.",
    )
    return parser.parse_args()


def load_tasks(json_path: Path) -> List[Dict[str, Any]]:
    if not json_path.exists():
        raise FileNotFoundError(f"Input json not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict):
        for key in ("data", "tasks", "records", "items", "RECORDS"):
            value = data.get(key)
            if isinstance(value, list):
                return value

        for key, value in data.items():
            if isinstance(value, list):
                if not value:
                    return value
                if all(isinstance(x, dict) for x in value):
                    print(f"[INFO] detected task list under top-level key: {key}")
                    return value

    raise ValueError(
        "Unsupported benchmark json format: expected a list or a dict containing a task list."
    )


def get_task_id(raw_task: Dict[str, Any]) -> str:
    return str(
        raw_task.get("_id")
        or raw_task.get("question_id")
        or raw_task.get("task_id")
        or raw_task.get("id")
        or ""
    ).strip()


def filter_tasks(
    tasks: List[Dict[str, Any]],
    task_id: Optional[str],
    start_index: int,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    if task_id:
        filtered = [task for task in tasks if get_task_id(task) == task_id]
        if not filtered:
            raise ValueError(f"Task id not found: {task_id}")
        return filtered

    if start_index < 0:
        raise ValueError("start_index must be >= 0")

    sliced = tasks[start_index:]
    if limit is not None:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        sliced = sliced[:limit]
    return sliced


def detect_lang_from_path(path: Path) -> str:
    name = path.name.lower()
    if "java" in name:
        return "java"
    if "python" in name or "py" in name:
        return "python"
    return "unknown"


def load_existing_results(output_path: Path) -> List[Dict[str, Any]]:
    if not output_path.exists():
        return []

    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Existing output json must be a list.")
    return data


def _result_merge_key(item: Dict[str, Any]) -> str:
    item_id = str(item.get("_id", "")).strip()
    lang = str(item.get("lang", "")).strip().lower()
    if not item_id:
        raise ValueError("Result item is missing _id.")
    return f"{lang}::{item_id}" if lang else item_id


def merge_results(
    existing_results: List[Dict[str, Any]],
    new_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}

    for item in existing_results:
        merged[_result_merge_key(item)] = item

    for item in new_results:
        merged[_result_merge_key(item)] = item

    ordered_keys: List[str] = []
    seen = set()

    for source in (existing_results, new_results):
        for item in source:
            key = _result_merge_key(item)
            if key not in seen:
                ordered_keys.append(key)
                seen.add(key)

    for key in merged.keys():
        if key not in seen:
            ordered_keys.append(key)
            seen.add(key)

    return [merged[key] for key in ordered_keys]


def save_results(
    output_path: Path,
    results: List[Dict[str, Any]],
    pretty: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if pretty:
            json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            json.dump(results, f, ensure_ascii=False)


def extract_answer_text(result: Dict[str, Any]) -> str:
    """
    Extract one code string from baseline pipeline output.
    Priority:
    1. benchmark-style generate_results[0]
    2. common flat string fields
    3. nested fallback
    """
    # 1) your actual benchmark format
    generate_results = result.get("generate_results")
    if isinstance(generate_results, list) and generate_results:
        first = generate_results[0]
        if isinstance(first, str):
            return first

    # 2) common flat fields
    for key in (
        "answer",
        "code",
        "prediction",
        "generated_code",
        "output",
        "response",
        "final_code",
    ):
        value = result.get(key)
        if isinstance(value, str):
            return value

    # 3) optional nested fallback
    for container_key in ("result", "data", "payload"):
        container = result.get(container_key)
        if isinstance(container, dict):
            nested_generate_results = container.get("generate_results")
            if isinstance(nested_generate_results, list) and nested_generate_results:
                first = nested_generate_results[0]
                if isinstance(first, str):
                    return first

    return ""


def run_task_passes(
    pipeline: BaselinePipeline,
    raw_task: Dict[str, Any],
    lang: str,
    num_passes: int,
) -> Dict[str, Any]:
    task_id = get_task_id(raw_task)
    if not task_id:
        raise ValueError("Task is missing id.")

    final_item: Dict[str, Any] = {
        "_id": task_id,
        "lang": lang,
    }

    for pass_idx in range(1, num_passes + 1):
        print(f"    - pass {pass_idx}/{num_passes}")
        result = pipeline.run_task_to_dict(raw_task)
        answer_text = extract_answer_text(result)
        final_item[f"answer_{pass_idx}"] = answer_text

    return final_item


def build_dual_language_task_list(
    python_json: Path,
    java_json: Path,
    python_task_id: Optional[str],
    java_task_id: Optional[str],
    python_start_index: int,
    java_start_index: int,
    take_n_each: int,
) -> List[Tuple[str, Dict[str, Any]]]:
    python_tasks_all = load_tasks(python_json)
    java_tasks_all = load_tasks(java_json)

    python_tasks = filter_tasks(
        tasks=python_tasks_all,
        task_id=python_task_id,
        start_index=python_start_index,
        limit=take_n_each,
    )
    java_tasks = filter_tasks(
        tasks=java_tasks_all,
        task_id=java_task_id,
        start_index=java_start_index,
        limit=take_n_each,
    )

    pairs: List[Tuple[str, Dict[str, Any]]] = []
    for task in python_tasks:
        pairs.append(("python", task))
    for task in java_tasks:
        pairs.append(("java", task))
    return pairs


def main() -> None:
    args = parse_args()
    cfg = load_config()

    output_json = Path(args.output_json or cfg.run.output_json_path)
    num_passes = args.num_passes if args.num_passes is not None else cfg.run.num_passes

    if num_passes <= 0:
        raise ValueError("num_passes must be >= 1")

    generator = BaselineGenerator(cfg)
    pipeline = BaselinePipeline(generator=generator)

    new_results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Mode A: dual-language mode (python + java)
    # ------------------------------------------------------------------
    if args.python_json or args.java_json:
        python_json = Path(args.python_json or cfg.run.python_input_json_path)
        java_json = Path(args.java_json or cfg.run.java_input_json_path)

        selected_items = build_dual_language_task_list(
            python_json=python_json,
            java_json=java_json,
            python_task_id=args.python_task_id,
            java_task_id=args.java_task_id,
            python_start_index=args.python_start_index,
            java_start_index=args.java_start_index,
            take_n_each=args.take_n_each,
        )

        print("=" * 100)
        print("Baseline CoderEval runner (dual-language mode)")
        print(f"python_json    : {python_json}")
        print(f"java_json      : {java_json}")
        print(f"output_json    : {output_json}")
        print(f"take_n_each    : {args.take_n_each}")
        print(f"num_passes     : {num_passes}")
        print(f"selected_total : {len(selected_items)}")
        print("=" * 100)

        total = len(selected_items)
        for i, (lang, raw_task) in enumerate(selected_items, start=1):
            task_id = get_task_id(raw_task)
            print(f"[{i}/{total}] running {lang} task: {task_id}")

            try:
                result = run_task_passes(
                    pipeline=pipeline,
                    raw_task=raw_task,
                    lang=lang,
                    num_passes=num_passes,
                )
                new_results.append(result)
                print(f"[OK] lang={lang} task={task_id}")
            except Exception as exc:
                print(f"[ERROR] lang={lang} task={task_id} error={exc}")

    # ------------------------------------------------------------------
    # Mode B: legacy single-file mode
    # ------------------------------------------------------------------
    else:
        input_json = Path(args.input_json or cfg.run.input_json_path)
        start_index = args.start_index if args.start_index is not None else cfg.run.start_index
        limit = args.limit if args.limit is not None else cfg.run.task_limit

        all_tasks = load_tasks(input_json)
        selected_tasks = filter_tasks(
            tasks=all_tasks,
            task_id=args.task_id,
            start_index=start_index,
            limit=limit,
        )
        lang = detect_lang_from_path(input_json)

        print("=" * 100)
        print("Baseline CoderEval runner (single-file mode)")
        print(f"input_json   : {input_json}")
        print(f"output_json  : {output_json}")
        print(f"task_id      : {args.task_id or '<batch>'}")
        print(f"start_index  : {start_index}")
        print(f"limit        : {limit}")
        print(f"num_passes   : {num_passes}")
        print(f"selected     : {len(selected_tasks)}")
        print("=" * 100)

        total = len(selected_tasks)
        for i, raw_task in enumerate(selected_tasks, start=1):
            task_id = get_task_id(raw_task)
            print(f"[{i}/{total}] running {lang} task: {task_id}")

            try:
                result = run_task_passes(
                    pipeline=pipeline,
                    raw_task=raw_task,
                    lang=lang,
                    num_passes=num_passes,
                )
                new_results.append(result)
                print(f"[OK] lang={lang} task={task_id}")
            except Exception as exc:
                print(f"[ERROR] lang={lang} task={task_id} error={exc}")

    if args.overwrite:
        final_results = new_results
    else:
        existing_results = load_existing_results(output_json)
        final_results = merge_results(existing_results, new_results)

    save_results(output_json, final_results, pretty=args.pretty)

    print("-" * 100)
    print(f"new_results      : {len(new_results)}")
    print(f"final_json_items : {len(final_results)}")
    print(f"saved_to         : {output_json}")
    print("-" * 100)


if __name__ == "__main__":
    main()