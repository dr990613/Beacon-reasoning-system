# run_codereval_baseline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        help="Path to benchmark json file. Overrides config if provided.",
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
        help="Run only one task by id (_id/question_id/task_id/id).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N tasks after filtering.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="Start index for batch execution.",
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
        # 先处理常见字段
        for key in ("data", "tasks", "records", "items", "RECORDS"):
            value = data.get(key)
            if isinstance(value, list):
                return value

        # 再做一层宽松匹配：只要某个 value 是 list[dict]，就用它
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


def load_existing_results(output_path: Path) -> List[Dict[str, Any]]:
    if not output_path.exists():
        return []

    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Existing output json must be a list.")
    return data


def merge_results(
    existing_results: List[Dict[str, Any]],
    new_results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    以 _id 为键更新结果。
    新结果覆盖旧结果，未涉及的旧结果保留。
    """
    merged: Dict[str, Dict[str, Any]] = {}

    for item in existing_results:
        item_id = str(item.get("_id", "")).strip()
        if item_id:
            merged[item_id] = item

    for item in new_results:
        item_id = str(item.get("_id", "")).strip()
        if not item_id:
            raise ValueError("Generated result is missing _id.")
        merged[item_id] = item

    # 保持稳定顺序：先按 existing/new 的出现顺序，再补剩余
    ordered_ids: List[str] = []
    seen = set()

    for source in (existing_results, new_results):
        for item in source:
            item_id = str(item.get("_id", "")).strip()
            if item_id and item_id not in seen:
                ordered_ids.append(item_id)
                seen.add(item_id)

    for item_id in merged.keys():
        if item_id not in seen:
            ordered_ids.append(item_id)
            seen.add(item_id)

    return [merged[item_id] for item_id in ordered_ids]


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


def main() -> None:
    args = parse_args()
    cfg = load_config()

    input_json = Path(args.input_json or cfg.run.input_json_path)
    output_json = Path(args.output_json or cfg.run.output_json_path)
    start_index = args.start_index if args.start_index is not None else cfg.run.start_index
    limit = args.limit if args.limit is not None else cfg.run.task_limit

    all_tasks = load_tasks(input_json)
    selected_tasks = filter_tasks(
        tasks=all_tasks,
        task_id=args.task_id,
        start_index=start_index,
        limit=limit,
    )

    generator = BaselineGenerator(cfg)
    pipeline = BaselinePipeline(generator=generator)

    print("=" * 100)
    print("Baseline CoderEval runner")
    print(f"input_json   : {input_json}")
    print(f"output_json  : {output_json}")
    print(f"task_id      : {args.task_id or '<batch>'}")
    print(f"start_index  : {start_index}")
    print(f"limit        : {limit}")
    print(f"selected     : {len(selected_tasks)}")
    print("=" * 100)

    new_results: List[Dict[str, Any]] = []
    total = len(selected_tasks)

    for i, raw_task in enumerate(selected_tasks, start=1):
        task_id = get_task_id(raw_task)
        print(f"[{i}/{total}] running task: {task_id}")

        try:
            result = pipeline.run_task_to_dict(raw_task)
            new_results.append(result)
            print(f"[OK] task={task_id}")
        except Exception as exc:
            print(f"[ERROR] task={task_id} error={exc}")

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