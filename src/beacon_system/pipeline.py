# src/beacon_system/pipeline.py
# -*- coding: utf-8 -*-

"""
Pipeline (唯一 orchestrator)

Fixed order (NO registry, NO branching explosion):
1) TaskAdapter.build_task() -> (TaskObject, ProjectIndex)
2) logic.engine.build(...) -> (BeaconIR, Constraints, debug?)
3) Generator.generate(...) -> code
4) Verifier.check(code, Constraints) -> report (optional)
5) RuntimeAdapter.run(task, patch) -> ExecutionResult
6) io.write_artifacts(...) every round

Hard rules:
- Only depend on adapters/base.py interfaces (TaskAdapter, RuntimeAdapter)
- No reasoning or constraint compilation here (handled by logic)
- Determinism: artifacts should be stable and replayable
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .adapters.base import TaskAdapter, RuntimeAdapter
from .llm.client import LLMClient
from .agents import generator as gen_mod
from .agents import verifier as ver_mod
from .logic.engine import build as logic_build
from .io import make_run_id, write_artifacts

from .types import (
    TaskObject,
    ProjectIndex,
    ReaderConfig,
    RunConfig,
    BeaconIR,
    Constraints,
    VerifierReport,
    ExecutionResult,
)


def _build_reader_config(reader_dict: Dict[str, Any]) -> ReaderConfig:
    """
    Backward compatible mapping:
    - New keys: validation_filter, max_local_nodes, max_global_inline
    - Legacy keys: enable_val_filter, max_nodes, max_depth (ignored or mapped)
    """
    # new -> primary
    validation_filter = reader_dict.get("validation_filter", None)
    if validation_filter is None:
        # legacy fallback
        validation_filter = reader_dict.get("enable_val_filter", True)

    max_local_nodes = reader_dict.get("max_local_nodes", None)
    if max_local_nodes is None:
        # legacy fallback: max_nodes was used as "reduce cap"
        max_local_nodes = reader_dict.get("max_nodes", None)

    return ReaderConfig(
        enable_global=bool(reader_dict.get("enable_global", True)),
        validation_filter=bool(validation_filter),
        max_local_nodes=(int(max_local_nodes) if max_local_nodes is not None else None),
        max_global_inline=(
            int(reader_dict["max_global_inline"]) if reader_dict.get("max_global_inline", None) is not None else None
        ),
    )

def _build_run_config(run_cfg_dict: Dict[str, Any], reader_cfg: ReaderConfig, model_cfg: Any) -> RunConfig:
    run_dict = dict(run_cfg_dict.get("run") or {})
    adapter_dict = dict(run_cfg_dict.get("adapter") or {})

    return RunConfig(
        seed=int(run_dict.get("seed", 0)),
        max_rounds=int(run_dict.get("max_rounds", 2)),
        use_verifier=bool(run_dict.get("use_verifier", True)),
        outputs_dir=str(run_dict.get("outputs_dir", "outputs/runs")),
        reader=reader_cfg,
        model=model_cfg,
        adapter=adapter_dict,  # snapshot-ish input, final snapshot from adapter.snapshot()
    )


def run(
    *,
    run_cfg_dict: Dict[str, Any],
    task_adapter: TaskAdapter,
    runtime: RuntimeAdapter,
    llm: LLMClient,
    memory: Optional[object] = None,
) -> None:
    """
    Execute one task end-to-end, possibly with Generate->Verify->Revise loops.

    memory is optional and treated as a passive object passed to generator.
    (Working memory materialization is always done by io.write_artifacts.)
    """
    reader_cfg = _build_reader_config(run_cfg_dict.get("reader") or {})
    run_cfg = _build_run_config(run_cfg_dict, reader_cfg, llm.cfg)

    run_id = make_run_id()
    run_dir = f"{run_cfg.outputs_dir.rstrip('/')}/{run_id}"

    # 1) Build task + index
    task, project_index = task_adapter.build_task()
    adapter_snapshot = {
        "task_adapter": task_adapter.snapshot(),
        "runtime": runtime.snapshot(),
    }

    # 2) Beacon build (Dual Outputs)
    build_res = logic_build(
        task=task,
        project_index=project_index,
        config=run_cfg.reader,
        seed=run_cfg.seed,
        with_debug=False,
    )
    ir = build_res.ir
    constraints = build_res.constraints

    # 3+) Rounds
    prev_code: Optional[str] = None
    report: Optional[VerifierReport] = None
    exec_result: Optional[ExecutionResult] = None

    for round_id in range(1, run_cfg.max_rounds + 1):
        if round_id == 1 or not prev_code:
            code = gen_mod.generate(task, ir, constraints, llm, memory=memory)
        else:
            directives = tuple(report.directives) if (report is not None) else tuple()
            code = gen_mod.revise(task, ir, constraints, llm, directives, prev_code, memory=memory)

        # 4) Verify (optional)
        report = None
        if run_cfg.use_verifier:
            report = ver_mod.check(code, constraints)

        # Persist artifacts for this round before running (so we can replay even if runtime crashes)
        write_artifacts(
            run_dir=run_dir,
            config=run_cfg,
            adapter_snapshot=adapter_snapshot,
            task=task,
            ir=ir,
            constraints=constraints,
            code=code,
            report=report,
            exec_result=None,
            round_id=round_id,
        )

        # If verifier failed, go next round to revise (no runtime run)
        if run_cfg.use_verifier and report is not None and not report.ok:
            prev_code = code
            continue

        # 5) Runtime run
        patch = {
            "target_file": task.target.get("file"),
            "target_qualname": task.target.get("qualname"),
            "new_code": code,
        }
        exec_result = runtime.run(task, patch)

        # Persist runtime artifacts
        write_artifacts(
            run_dir=run_dir,
            config=run_cfg,
            adapter_snapshot=adapter_snapshot,
            task=task,
            ir=ir,
            constraints=constraints,
            code=code,
            report=report,
            exec_result=exec_result,
            round_id=round_id,
        )

        # Stop early on success
        if exec_result is not None and exec_result.status == "pass":
            break

        prev_code = code