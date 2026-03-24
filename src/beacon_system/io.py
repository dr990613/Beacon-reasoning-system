# -*- coding: utf-8 -*-

"""
Artifact IO for Beacon system.

Design goals:
- Simple and stable artifact persistence
- JSON-first for reproducibility
- No business logic, only filesystem + serialization
- Schema-tolerant: works with dataclass / dict / plain objects
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
import json


def ensure_dir(path: Path | str) -> Path:
    """Ensure directory exists and return Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def utc_now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(timezone.utc).isoformat()


def to_jsonable(obj: Any) -> Any:
    """
    Convert common Python objects into JSON-serializable structures.

    Supports:
    - None / primitive types
    - dict / list / tuple / set
    - dataclass instances
    - pathlib.Path
    - objects with __dict__
    - fallback to str(obj)
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if is_dataclass(obj):
        return to_jsonable(asdict(obj))

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(x) for x in obj]

    if hasattr(obj, "__dict__"):
        return to_jsonable(vars(obj))

    return str(obj)


def dump_json(path: Path | str, data: Any, *, indent: int = 2) -> Path:
    """Write JSON data to path."""
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, ensure_ascii=False, indent=indent, sort_keys=True)
    return p


def dump_text(path: Path | str, text: str) -> Path:
    """Write text to path."""
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8")
    return p


def load_json(path: Path | str, default: Optional[Any] = None) -> Any:
    """Load JSON file, return default if path not exists."""
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_task_dir(output_dir: Path | str, task_id: str) -> Path:
    """
    Create stable task artifact directory:
    <output_dir>/<task_id>/
    """
    root = ensure_dir(output_dir)
    return ensure_dir(root / str(task_id))


def save_logic_artifacts(
    output_dir: Path | str,
    task_id: str,
    logic_result: Any,
) -> Dict[str, str]:
    """
    Save logic-related artifacts.

    Outputs:
    - logic_result.json
    - raw_ir.json (if present)
    - beacon_tree.json (if present)
    - signature_hints.json (if present)
    - constraint_summary.json (if present)
    """
    task_dir = make_task_dir(output_dir, task_id)
    logic_dir = ensure_dir(task_dir / "logic")

    result_dict = to_jsonable(logic_result)
    paths: Dict[str, str] = {}

    paths["logic_result"] = str(dump_json(logic_dir / "logic_result.json", result_dict))

    if isinstance(result_dict, dict):
        for key in ("raw_ir", "beacon_tree", "signature_hints", "constraint_summary", "debug"):
            if key in result_dict:
                paths[key] = str(dump_json(logic_dir / f"{key}.json", result_dict[key]))

    return paths


def save_generation_artifacts(
    output_dir: Path | str,
    task_id: str,
    generation_result: Any,
    *,
    round_name: str = "round_1",
) -> Dict[str, str]:
    """
    Save generation artifacts.

    Outputs:
    - generation/<round_name>/generation_result.json
    - generation/<round_name>/generated_code.py|txt
    - generation/<round_name>/prompt_snapshot.txt
    """
    task_dir = make_task_dir(output_dir, task_id)
    gen_dir = ensure_dir(task_dir / "generation" / round_name)

    result_dict = to_jsonable(generation_result)
    paths: Dict[str, str] = {}
    paths["generation_result"] = str(dump_json(gen_dir / "generation_result.json", result_dict))

    if isinstance(result_dict, dict):
        code = result_dict.get("generated_code")
        prompt = result_dict.get("prompt_snapshot")
        raw_response = result_dict.get("raw_response")

        if isinstance(code, str):
            paths["generated_code"] = str(dump_text(gen_dir / "generated_code.py", code))
        if isinstance(prompt, str):
            paths["prompt_snapshot"] = str(dump_text(gen_dir / "prompt_snapshot.txt", prompt))
        if raw_response is not None:
            paths["raw_response"] = str(dump_json(gen_dir / "raw_response.json", raw_response))

    return paths


def save_verification_artifacts(
    output_dir: Path | str,
    task_id: str,
    verification_result: Any,
    *,
    round_name: str = "round_1",
) -> Dict[str, str]:
    """
    Save verification artifacts.

    Outputs:
    - verification/<round_name>/verification_result.json
    """
    task_dir = make_task_dir(output_dir, task_id)
    ver_dir = ensure_dir(task_dir / "verification" / round_name)

    result_dict = to_jsonable(verification_result)
    paths: Dict[str, str] = {}
    paths["verification_result"] = str(dump_json(ver_dir / "verification_result.json", result_dict))
    return paths


def save_run_trace(
    output_dir: Path | str,
    task_id: str,
    run_trace: Any,
) -> str:
    """
    Save full run trace to:
    <output_dir>/<task_id>/run_trace.json
    """
    task_dir = make_task_dir(output_dir, task_id)
    trace_dict = to_jsonable(run_trace)

    if isinstance(trace_dict, dict) and "metadata" not in trace_dict:
        trace_dict["metadata"] = {"saved_at_utc": utc_now_iso()}
    elif isinstance(trace_dict, dict):
        trace_dict.setdefault("metadata", {})
        trace_dict["metadata"].setdefault("saved_at_utc", utc_now_iso())

    return str(dump_json(task_dir / "run_trace.json", trace_dict))