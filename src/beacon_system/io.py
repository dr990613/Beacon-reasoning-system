# src/beacon_system/io.py
# -*- coding: utf-8 -*-

"""
Artifacts I/O (working-memory materialization)

- Deterministic JSON serialization: stable_json(obj)
- Run folder layout: outputs/runs/<run_id>/
- Persist pipeline / workflow artifacts for replay and regression debugging
- Centralize all artifact writes here; other modules should not write ad hoc files
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import time
from typing import Any, Dict, Optional, Sequence

import yaml

from .types import (
    BeaconIR,
    BeaconUsageReport,
    Constraints,
    ExecResult,
    FormatValidationResult,
    GenerationPayload,
    MemoryReadResult,
    MemoryWriteResult,
    RunConfig,
    TaskObject,
    ThoughtCandidate,
    ThoughtScore,
    VerifierResult,
)


def make_run_id() -> str:
    """
    Run ID: UTC-ish timestamp + short hash suffix.

    Deterministic is NOT required here; uniqueness is enough.
    """
    ts = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    suffix = hashlib.sha1(f"{time.time_ns()}".encode("utf-8")).hexdigest()[:8]
    return f"{ts}-{suffix}"


def _to_primitive(obj: Any) -> Any:
    """
    Convert dataclasses / tuples / sets into JSON-serializable primitives.
    Keep it minimal and deterministic.
    """
    if obj is None:
        return None

    if dataclasses.is_dataclass(obj):
        return {k: _to_primitive(v) for k, v in dataclasses.asdict(obj).items()}

    if isinstance(obj, dict):
        return {str(k): _to_primitive(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_to_primitive(x) for x in obj]

    if isinstance(obj, set):
        return sorted(
            [_to_primitive(x) for x in obj],
            key=lambda x: json.dumps(x, ensure_ascii=False, sort_keys=True),
        )

    if isinstance(obj, (str, int, float, bool)):
        return obj

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
    parent = os.path.dirname(path)
    if parent:
        _ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_json(path: str, obj: Any) -> None:
    _write_text(path, stable_json(obj) + "\n")


def _write_yaml(path: str, obj: Any) -> None:
    prim = _to_primitive(obj)
    parent = os.path.dirname(path)
    if parent:
        _ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(prim, f, sort_keys=True, allow_unicode=True)


def ensure_run_dir(outputs_dir: str, run_id: str) -> str:
    """
    Create and return outputs/runs/<run_id>-style directory.
    """
    run_dir = os.path.join(outputs_dir, run_id)
    _ensure_dir(run_dir)
    return run_dir


def write_json_artifact(run_dir: str, filename: str, obj: Any) -> str:
    """
    Generic JSON artifact writer.
    """
    path = os.path.join(run_dir, filename)
    _write_json(path, obj)
    return path


def write_text_artifact(run_dir: str, filename: str, text: str) -> str:
    """
    Generic text artifact writer.
    """
    path = os.path.join(run_dir, filename)
    _write_text(path, text)
    return path


def write_static_artifacts(
    run_dir: str,
    *,
    config: RunConfig,
    adapter_snapshot: Dict[str, Any],
    task: TaskObject,
    ir: BeaconIR,
    constraints: Constraints,
) -> None:
    """
    Persist run-level static artifacts.

    These files are usually written once per run, but overwrite is harmless.
    """
    _ensure_dir(run_dir)
    _write_yaml(os.path.join(run_dir, "config.yaml"), config)
    _write_json(os.path.join(run_dir, "adapter_snapshot.json"), adapter_snapshot)
    _write_json(os.path.join(run_dir, "task.json"), task)
    _write_json(os.path.join(run_dir, "ir.json"), ir)
    _write_json(os.path.join(run_dir, "constraints.json"), constraints)


def write_thoughts(
    run_dir: str,
    *,
    thoughts: Sequence[ThoughtCandidate],
    round_id: int,
) -> None:
    """
    Persist planning thoughts for one round.
    """
    _write_json(os.path.join(run_dir, f"thoughts_round{round_id}.json"), list(thoughts))


def write_scores(
    run_dir: str,
    *,
    scores: Sequence[ThoughtScore],
    round_id: int,
) -> None:
    """
    Persist thought scores for one round.
    """
    _write_json(os.path.join(run_dir, f"scores_round{round_id}.json"), list(scores))


def write_generation(
    run_dir: str,
    *,
    generation: Optional[GenerationPayload],
    round_id: int,
    default_ext: str = ".py",
) -> None:
    """
    Persist generation payload and primary code block.

    Files:
    - generation_round{n}.json
    - code_round{n}{ext}
    """
    if generation is None:
        return

    _write_json(os.path.join(run_dir, f"generation_round{round_id}.json"), generation)

    primary = generation.primary
    code = (primary.content or "").rstrip() + "\n"
    ext = default_ext
    if primary.filename:
        _, maybe_ext = os.path.splitext(primary.filename)
        if maybe_ext:
            ext = maybe_ext

    _write_text(os.path.join(run_dir, f"code_round{round_id}{ext}"), code)

    if generation.raw_text:
        _write_text(
            os.path.join(run_dir, f"raw_generation_round{round_id}.txt"),
            generation.raw_text.rstrip() + "\n",
        )


def write_format_check(
    run_dir: str,
    *,
    format_check: Optional[FormatValidationResult],
    round_id: int,
    default_ext: str = ".py",
) -> None:
    """
    Persist format validation result.

    If normalized_code exists, also write it as a text artifact for inspection.
    """
    if format_check is None:
        return

    _write_json(os.path.join(run_dir, f"format_check_round{round_id}.json"), format_check)

    if format_check.normalized_code:
        _write_text(
            os.path.join(run_dir, f"normalized_code_round{round_id}{default_ext}"),
            format_check.normalized_code.rstrip() + "\n",
        )


def write_verifier(
    run_dir: str,
    *,
    report: Optional[VerifierResult],
    round_id: int,
) -> None:
    """
    Persist verifier result for one round.
    """
    if report is None:
        return
    _write_json(os.path.join(run_dir, f"verifier_round{round_id}.json"), report)


def write_beacon_usage(
    run_dir: str,
    *,
    usage: Optional[BeaconUsageReport],
    round_id: int,
) -> None:
    """
    Persist Beacon usage check result for one round.
    """
    if usage is None:
        return
    _write_json(os.path.join(run_dir, f"beacon_usage_round{round_id}.json"), usage)


def write_exec_result(
    run_dir: str,
    *,
    exec_result: Optional[ExecResult],
    round_id: int,
) -> None:
    """
    Persist execution result for one round.

    Files:
    - exec_round{n}.json
    - exec_round{n}.trace.txt
    - metrics_round{n}.json
    """
    if exec_result is None:
        return

    _write_json(os.path.join(run_dir, f"exec_round{round_id}.json"), exec_result)

    if exec_result.trace:
        _write_text(
            os.path.join(run_dir, f"exec_round{round_id}.trace.txt"),
            exec_result.trace.rstrip() + "\n",
        )

    if exec_result.metrics:
        _write_json(os.path.join(run_dir, f"metrics_round{round_id}.json"), exec_result.metrics)


def write_memory_artifacts(
    run_dir: str,
    *,
    memory_read: Optional[MemoryReadResult] = None,
    memory_write: Optional[MemoryWriteResult] = None,
) -> None:
    """
    Persist memory read/write artifacts.
    """
    if memory_read is not None:
        _write_json(os.path.join(run_dir, "memory_read.json"), memory_read)

    if memory_write is not None:
        _write_json(os.path.join(run_dir, "memory_write.json"), memory_write)


def write_artifacts(
    run_dir: str,
    *,
    config: RunConfig,
    adapter_snapshot: Dict[str, Any],
    task: TaskObject,
    ir: BeaconIR,
    constraints: Constraints,
    round_id: int,
    generation: Optional[GenerationPayload] = None,
    format_check: Optional[FormatValidationResult] = None,
    report: Optional[VerifierResult] = None,
    beacon_usage: Optional[BeaconUsageReport] = None,
    exec_result: Optional[ExecResult] = None,
    thoughts: Optional[Sequence[ThoughtCandidate]] = None,
    scores: Optional[Sequence[ThoughtScore]] = None,
    memory_read: Optional[MemoryReadResult] = None,
    memory_write: Optional[MemoryWriteResult] = None,
    default_code_ext: str = ".py",
) -> None:
    """
    Unified artifact writer for one workflow round.

    Static files:
    - config.yaml
    - adapter_snapshot.json
    - task.json
    - ir.json
    - constraints.json

    Optional per-round files:
    - thoughts_round{n}.json
    - scores_round{n}.json
    - generation_round{n}.json
    - code_round{n}.py
    - raw_generation_round{n}.txt
    - format_check_round{n}.json
    - normalized_code_round{n}.py
    - verifier_round{n}.json
    - beacon_usage_round{n}.json
    - exec_round{n}.json
    - exec_round{n}.trace.txt
    - metrics_round{n}.json

    Optional run-level memory files:
    - memory_read.json
    - memory_write.json
    """
    _ensure_dir(run_dir)

    write_static_artifacts(
        run_dir,
        config=config,
        adapter_snapshot=adapter_snapshot,
        task=task,
        ir=ir,
        constraints=constraints,
    )

    if thoughts is not None:
        write_thoughts(run_dir, thoughts=thoughts, round_id=round_id)

    if scores is not None:
        write_scores(run_dir, scores=scores, round_id=round_id)

    write_generation(
        run_dir,
        generation=generation,
        round_id=round_id,
        default_ext=default_code_ext,
    )

    write_format_check(
        run_dir,
        format_check=format_check,
        round_id=round_id,
        default_ext=default_code_ext,
    )

    write_verifier(run_dir, report=report, round_id=round_id)
    write_beacon_usage(run_dir, usage=beacon_usage, round_id=round_id)
    write_exec_result(run_dir, exec_result=exec_result, round_id=round_id)
    write_memory_artifacts(
        run_dir,
        memory_read=memory_read,
        memory_write=memory_write,
    )