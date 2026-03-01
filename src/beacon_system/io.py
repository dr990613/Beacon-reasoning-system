# src/beacon_system/io.py
# -*- coding: utf-8 -*-

"""
Artifacts I/O (Working memory materialization)

- Deterministic JSON serialization: stable_json(obj)
- Run folder layout: outputs/runs/<run_id>/
- Persist per-round artifacts for replay and regression debugging
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
import hashlib
from typing import Any, Dict, Optional

import yaml

from .types import (
    RunConfig,
    TaskObject,
    BeaconIR,
    Constraints,
    VerifierReport,
    ExecutionResult,
)


def make_run_id() -> str:
    """
    Run ID: UTC-ish timestamp + short hash suffix.
    Deterministic is NOT required here; uniqueness is.
    """
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    suffix = hashlib.sha1(f"{time.time_ns()}".encode("utf-8")).hexdigest()[:8]
    return f"{ts}-{suffix}"


def _to_primitive(obj: Any) -> Any:
    """
    Convert dataclasses / tuples / sets into JSON-serializable primitives.
    Keep it minimal and deterministic (sorting where needed).
    """
    if obj is None:
        return None

    if dataclasses.is_dataclass(obj):
        return {k: _to_primitive(v) for k, v in dataclasses.asdict(obj).items()}

    if isinstance(obj, dict):
        # sort keys deterministically in stable_json (json.dumps sort_keys=True),
        # here just ensure values are primitive
        return {str(k): _to_primitive(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_primitive(x) for x in obj]

    if isinstance(obj, set):
        return sorted([_to_primitive(x) for x in obj], key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True))

    # Basic types
    if isinstance(obj, (str, int, float, bool)):
        return obj

    # Fallback: represent unknown objects safely
    return {"repr": repr(obj)}


def stable_json(obj: Any) -> str:
    """
    Deterministic JSON string:
    - ensure_ascii=False for readability
    - sort_keys=True for stable ordering
    - separators for stable whitespace
    """
    prim = _to_primitive(obj)
    return json.dumps(prim, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_json(path: str, obj: Any) -> None:
    _write_text(path, stable_json(obj) + "\n")


def _write_yaml(path: str, obj: Any) -> None:
    # YAML is for human inspection; determinism not critical, but keep stable where easy.
    prim = _to_primitive(obj)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(prim, f, sort_keys=True, allow_unicode=True)


def write_artifacts(
    run_dir: str,
    *,
    config: RunConfig,
    adapter_snapshot: Dict[str, Any],
    task: TaskObject,
    ir: BeaconIR,
    constraints: Constraints,
    code: str,
    report: Optional[VerifierReport],
    exec_result: Optional[ExecutionResult],
    round_id: int,
) -> None:
    """
    Persist working-memory artifacts for this round.

    File naming (minimal):
    - config.yaml
    - adapter_snapshot.json
    - task.json
    - ir.json
    - constraints.json
    - code_round{n}.py
    - verifier_round{n}.json (if report)
    - exec_round{n}.json (if exec_result)
    - exec_round{n}.trace.txt (if exec_result.trace)
    - metrics_round{n}.json (if exec_result.metrics)
    """
    _ensure_dir(run_dir)

    # Write “static” files once (idempotent overwrite is fine)
    _write_yaml(os.path.join(run_dir, "config.yaml"), config)
    _write_json(os.path.join(run_dir, "adapter_snapshot.json"), adapter_snapshot)
    _write_json(os.path.join(run_dir, "task.json"), task)
    _write_json(os.path.join(run_dir, "ir.json"), ir)
    _write_json(os.path.join(run_dir, "constraints.json"), constraints)

    # Per-round
    _write_text(os.path.join(run_dir, f"code_round{round_id}.py"), (code or "").rstrip() + "\n")

    if report is not None:
        _write_json(os.path.join(run_dir, f"verifier_round{round_id}.json"), report)

    if exec_result is not None:
        _write_json(os.path.join(run_dir, f"exec_round{round_id}.json"), exec_result)
        if exec_result.trace:
            _write_text(os.path.join(run_dir, f"exec_round{round_id}.trace.txt"), exec_result.trace.rstrip() + "\n")
        if exec_result.metrics:
            _write_json(os.path.join(run_dir, f"metrics_round{round_id}.json"), exec_result.metrics)