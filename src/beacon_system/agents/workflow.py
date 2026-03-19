# src/beacon_system/agents/workflow.py
# -*- coding: utf-8 -*-

"""
Agent workflow orchestration.

Scope:
- Orchestrate:
  prepare -> logic -> checks -> memory -> plan -> score -> generate -> verify -> revise
- Keep logic as the mandatory reasoning tool node.
- Keep runtime execution OUTSIDE this module.
- Return structured intermediate results for pipeline/io to persist.

Non-goals:
- no runtime execution here
- no artifact writing here
- no env access here
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple
import inspect

from ..llm.client import LLMClient
from ..types import (
    AgentConfig,
    BeaconUsageReport,
    BuildResult,
    Constraints,
    FormatValidationResult,
    GenerationPayload,
    GenerationRoundResult,
    MemoryReadResult,
    MemoryWriteResult,
    ProjectIndex,
    ReaderConfig,
    RunConfig,
    TaskObject,
    ThoughtCandidate,
    ThoughtScore,
    VerifierResult,
)
from ..logic import engine as logic_engine
from .checks import check_beacon_usage, check_logic_outputs
from .generator import generate_code, revise_code
from .memory import ExperienceMemory
from .planning import generate_thoughts
from .scoring import score_thoughts, select_top_thoughts
from .verifier import verify_code


def _safe_text(text: Optional[str]) -> str:
    return str(text or "").strip()


def _to_plain(obj: Any) -> Any:
    if obj is None:
        return None
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, dict):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(x) for x in obj]
    return obj


def _build_result_from_any(raw: Any) -> BuildResult:
    """
    Accept a few possible logic.engine.build(...) return shapes:
    - BuildResult dataclass
    - any object with .ir and .constraints attributes
    - (ir, constraints)
    - (ir, constraints, debug)
    - {"ir": ..., "constraints": ..., "debug": ...}
    """
    if raw is None:
        raise TypeError("logic.engine.build returned None")

    # 1) exact shared BuildResult
    if isinstance(raw, BuildResult):
        return raw

    # 2) duck-typed object with attributes: .ir / .constraints / optional .debug
    if hasattr(raw, "ir") and hasattr(raw, "constraints"):
        return BuildResult(
            ir=getattr(raw, "ir"),
            constraints=getattr(raw, "constraints"),
            debug=getattr(raw, "debug", None),
        )

    # 3) tuple forms
    if isinstance(raw, tuple):
        if len(raw) == 2:
            return BuildResult(ir=raw[0], constraints=raw[1], debug=None)
        if len(raw) >= 3:
            return BuildResult(ir=raw[0], constraints=raw[1], debug=raw[2])

    # 4) dict form
    if isinstance(raw, dict) and "ir" in raw and "constraints" in raw:
        return BuildResult(
            ir=raw["ir"],
            constraints=raw["constraints"],
            debug=raw.get("debug"),
        )

    raise TypeError(
        f"logic.engine.build returned unsupported shape: {type(raw)!r}"
    )


def _call_logic_build(
    *,
    task: TaskObject,
    project_index: ProjectIndex,
    reader_config: ReaderConfig,
) -> BuildResult:
    """
    Compatibility wrapper for logic.engine.build(...).

    Adapt to different possible signatures by inspecting parameter names.
    Typical variants:
    - build(task, project_index, config)
    - build(task=..., project_index=..., config=...)
    - build(task=..., index=..., config=...)
    - build(task, index, config)
    """
    build_fn = logic_engine.build
    sig = inspect.signature(build_fn)
    params = sig.parameters

    kwargs: Dict[str, Any] = {}
    positional = []

    # Map task
    if "task" in params:
        kwargs["task"] = task
    else:
        positional.append(task)

    # Map project index aliases
    if "project_index" in params:
        kwargs["project_index"] = project_index
    elif "index" in params:
        kwargs["index"] = project_index
    elif "project" in params:
        kwargs["project"] = project_index
    else:
        positional.append(project_index)

    # Map config aliases
    if "config" in params:
        kwargs["config"] = reader_config
    elif "reader_config" in params:
        kwargs["reader_config"] = reader_config
    elif "reader" in params:
        kwargs["reader"] = reader_config
    else:
        positional.append(reader_config)

    # First try keyword-friendly call
    try:
        main_kwargs = {}

        if "task" in params:
            main_kwargs["task"] = task
        if "project_index" in params:
            main_kwargs["project_index"] = project_index
        if "config" in params:
            main_kwargs["config"] = reader_config
        if "seed" in params:
            main_kwargs["seed"] = 0
        if "with_debug" in params:
            main_kwargs["with_debug"] = True

        return _build_result_from_any(build_fn(**main_kwargs))
    except TypeError:
        pass

    # Fallback: strict positional call
    try:
        return _build_result_from_any(build_fn(task, project_index, reader_config))
    except TypeError:
        pass

    # Fallback: some old builds may accept only (task, config)
    try:
        return _build_result_from_any(build_fn(task, reader_config))
    except TypeError:
        pass

    # Fallback: some old builds may accept only keyword pairs
    fallback_attempts = [
        {"task": task, "project_index": project_index, "config": reader_config},
        {"task": task, "index": project_index, "config": reader_config},
        {"task": task, "project_index": project_index, "reader_config": reader_config},
        {"task": task, "project_index": project_index, "reader": reader_config},
        {"task": task, "config": reader_config},
    ]

    errors = []
    for attempt in fallback_attempts:
        filtered = {k: v for k, v in attempt.items() if k in params}
        if not filtered:
            continue
        try:
            return _build_result_from_any(build_fn(**filtered))
        except TypeError as e:
            errors.append(repr(e))
            continue

    raise RuntimeError(
        "logic.engine.build invocation failed. "
        f"Detected signature={sig}. "
        f"Tried multiple compatibility paths. "
        f"Errors={errors}"
    )

def _memory_summary(
    *,
    task: TaskObject,
    final_verifier: Optional[VerifierResult],
    final_beacon_usage: Optional[BeaconUsageReport],
    rounds: Sequence[GenerationRoundResult],
) -> str:
    pieces = [
        f"task={task.id}",
        f"target={task.target.get('file', '')}::{task.target.get('qualname', '')}",
        f"rounds={len(rounds)}",
    ]

    if final_verifier is not None:
        pieces.append(
            f"verifier_ok={final_verifier.ok}, violations={len(final_verifier.violations or ())}"
        )
    if final_beacon_usage is not None:
        pieces.append(
            f"beacon_usage_ok={final_beacon_usage.ok}, "
            f"missing_symbols={len(final_beacon_usage.missing_symbols or ())}, "
            f"missing_calls={len(final_beacon_usage.missing_calls or ())}"
        )

    return "; ".join(pieces)


def _status_for_memory(
    *,
    final_verifier: Optional[VerifierResult],
    final_beacon_usage: Optional[BeaconUsageReport],
    require_beacon_usage_check: bool,
) -> str:
    if final_verifier is None:
        return "failure"

    if not final_verifier.ok:
        return "partial"

    if require_beacon_usage_check:
        if final_beacon_usage is None or not final_beacon_usage.ok:
            return "partial"

    return "success"


def _select_primary_thought(
    selected: Sequence[ThoughtCandidate],
    thoughts: Sequence[ThoughtCandidate],
) -> Optional[ThoughtCandidate]:
    if selected:
        return selected[0]
    if thoughts:
        return thoughts[0]
    return None


@dataclass(frozen=True)
class AgentWorkflowResult:
    """
    Structured output of the agent workflow.

    This stays local to workflow.py to avoid expanding shared contracts too early.
    """
    build: BuildResult
    logic_acceptance: Any
    memory_read: Optional[MemoryReadResult] = None
    memory_prompt: str = ""
    thoughts: Tuple[ThoughtCandidate, ...] = ()
    scores: Tuple[ThoughtScore, ...] = ()
    selected_thoughts: Tuple[ThoughtCandidate, ...] = ()
    rounds: Tuple[GenerationRoundResult, ...] = ()
    final_generation: Optional[GenerationPayload] = None
    final_format_check: Optional[FormatValidationResult] = None
    final_verifier: Optional[VerifierResult] = None
    final_beacon_usage: Optional[BeaconUsageReport] = None
    memory_write: Optional[MemoryWriteResult] = None
    success: bool = False
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentWorkflow:
    """
    Main Beacon agent workflow.

    Boundary:
    - consumes task/project index/config + LLM client
    - produces structured workflow result
    - does not execute runtime
    """
    llm: LLMClient
    memory_store_path: str = "outputs/memory/experience.jsonl"
    print_io: bool = False

    def _print(self, message: str) -> None:
        if self.print_io:
            print(f"[AgentWorkflow] {message}")

    def _resolve_configs(
        self,
        *,
        run_config: Optional[RunConfig],
        reader_config: Optional[ReaderConfig],
        agent_config: Optional[AgentConfig],
    ) -> Tuple[ReaderConfig, AgentConfig]:
        if run_config is not None:
            return run_config.reader, run_config.agent

        if reader_config is None:
            reader_config = ReaderConfig()
        if agent_config is None:
            agent_config = AgentConfig()

        return reader_config, agent_config

    def run(
        self,
        *,
        task: TaskObject,
        project_index: ProjectIndex,
        run_id: str = "",
        run_config: Optional[RunConfig] = None,
        reader_config: Optional[ReaderConfig] = None,
        agent_config: Optional[AgentConfig] = None,
    ) -> AgentWorkflowResult:
        reader_cfg, agent_cfg = self._resolve_configs(
            run_config=run_config,
            reader_config=reader_config,
            agent_config=agent_config,
        )

        self._print(f"start workflow: task={task.id}")

        # --------------------------------------------------
        # 1) logic (mandatory tool node)
        # --------------------------------------------------
        build = _call_logic_build(
            task=task,
            project_index=project_index,
            reader_config=reader_cfg,
        )
        ir = build.ir
        constraints = build.constraints
        self._print("logic build done")

        # --------------------------------------------------
        # 2) logic acceptance checks
        # --------------------------------------------------
        logic_acceptance = check_logic_outputs(
            ir=ir,
            constraints=constraints,
            print_io=self.print_io,
        )
        self._print(f"logic acceptance ok={logic_acceptance.ok}")

        if agent_cfg.require_logic_acceptance and not logic_acceptance.ok:
            return AgentWorkflowResult(
                build=build,
                logic_acceptance=logic_acceptance,
                success=False,
                meta={
                    "stage": "logic_acceptance",
                    "stopped_early": True,
                },
            )

        # --------------------------------------------------
        # 3) memory read
        # --------------------------------------------------
        memory_read: Optional[MemoryReadResult] = None
        memory_prompt = ""

        if agent_cfg.use_memory:
            memory = ExperienceMemory(
                store_path=self.memory_store_path,
                print_io=self.print_io,
            )
            memory_read = memory.read(task=task, constraints=constraints)
            memory_prompt = memory.format_for_prompt(memory_read)
            self._print(
                f"memory read done items={len(memory_read.items or ())}"
            )

        # --------------------------------------------------
        # 4) planning
        # --------------------------------------------------
        thoughts = tuple(
            generate_thoughts(
                llm=self.llm,
                task=task,
                ir=ir,
                constraints=constraints,
                memory_text=memory_prompt,
                output_format=agent_cfg.output_format,
                max_thoughts=agent_cfg.max_thoughts,
                print_io=self.print_io,
            )
        )
        self._print(f"planning done thoughts={len(thoughts)}")

        # --------------------------------------------------
        # 5) scoring + pruning
        # --------------------------------------------------
        scores = tuple(
            score_thoughts(
                task=task,
                ir=ir,
                constraints=constraints,
                thoughts=thoughts,
                print_io=self.print_io,
            )
        )
        selected_thoughts = tuple(
            select_top_thoughts(
                thoughts=thoughts,
                scores=scores,
                keep_top_k=agent_cfg.keep_top_k,
                print_io=self.print_io,
            )
        )
        selected_thought = _select_primary_thought(selected_thoughts, thoughts)
        self._print(
            f"scoring done scores={len(scores)} selected={len(selected_thoughts)}"
        )

        # --------------------------------------------------
        # 6) generate -> verify -> usage check -> revise loop
        # --------------------------------------------------
        rounds: list[GenerationRoundResult] = []

        previous_code = ""
        previous_verifier: Optional[VerifierResult] = None
        previous_beacon_usage: Optional[BeaconUsageReport] = None

        final_generation: Optional[GenerationPayload] = None
        final_format_check: Optional[FormatValidationResult] = None
        final_verifier: Optional[VerifierResult] = None
        final_beacon_usage: Optional[BeaconUsageReport] = None

        max_rounds = max(1, int(agent_cfg.max_rounds))

        for round_index in range(1, max_rounds + 1):
            self._print(f"round={round_index}/{max_rounds}")

            if round_index == 1:
                generation, format_check = generate_code(
                    llm=self.llm,
                    task=task,
                    ir=ir,
                    constraints=constraints,
                    selected_thought=selected_thought,
                    output_format=agent_cfg.output_format,
                    print_io=self.print_io,
                )
            else:
                generation, format_check = revise_code(
                    llm=self.llm,
                    task=task,
                    ir=ir,
                    constraints=constraints,
                    selected_thought=selected_thought,
                    previous_code=previous_code,
                    verifier_summary=_to_plain(previous_verifier),
                    runtime_summary=None,
                    beacon_usage_summary=_to_plain(previous_beacon_usage),
                    output_format=agent_cfg.output_format,
                    print_io=self.print_io,
                )

            code = generation.primary.content if generation is not None else ""

            verifier_result: Optional[VerifierResult] = None
            if agent_cfg.use_verifier:
                verifier_result = verify_code(
                    code=code,
                    constraints=constraints,
                    print_io=self.print_io,
                )

            beacon_usage: Optional[BeaconUsageReport] = None
            if agent_cfg.require_beacon_usage_check:
                beacon_usage = check_beacon_usage(
                    code=code,
                    ir=ir,
                    constraints=constraints,
                    print_io=self.print_io,
                )

            directives = ()
            if verifier_result is not None:
                directives = verifier_result.directives or ()

            round_result = GenerationRoundResult(
                round_index=round_index,
                selected_thought_id=selected_thought.id if selected_thought is not None else "",
                generation=generation,
                format_check=format_check,
                verifier=verifier_result,
                beacon_usage=beacon_usage,
                exec_result=None,
                directives=tuple(directives),
                meta={
                    "stage": "generate" if round_index == 1 else "revise",
                },
            )
            rounds.append(round_result)

            final_generation = generation
            final_format_check = format_check
            final_verifier = verifier_result
            final_beacon_usage = beacon_usage

            previous_code = code
            previous_verifier = verifier_result
            previous_beacon_usage = beacon_usage

            verifier_ok = True if verifier_result is None else verifier_result.ok
            usage_ok = True
            if agent_cfg.require_beacon_usage_check:
                usage_ok = bool(beacon_usage is not None and beacon_usage.ok)

            if verifier_ok and usage_ok:
                self._print("workflow generation loop converged")
                break

        # --------------------------------------------------
        # 7) memory write
        # --------------------------------------------------
        memory_write: Optional[MemoryWriteResult] = None
        if agent_cfg.use_memory:
            memory = ExperienceMemory(
                store_path=self.memory_store_path,
                print_io=self.print_io,
            )
            memory_write = memory.write(
                task=task,
                constraints=constraints,
                run_id=run_id,
                status=_status_for_memory(
                    final_verifier=final_verifier,
                    final_beacon_usage=final_beacon_usage,
                    require_beacon_usage_check=agent_cfg.require_beacon_usage_check,
                ),
                summary=_memory_summary(
                    task=task,
                    final_verifier=final_verifier,
                    final_beacon_usage=final_beacon_usage,
                    rounds=rounds,
                ),
                selected_thought_id=selected_thought.id if selected_thought is not None else "",
                verifier_result=final_verifier,
                exec_result=None,
                used_required_symbols=(
                    final_beacon_usage.used_required_symbols
                    if final_beacon_usage is not None else ()
                ),
                used_required_calls=(
                    final_beacon_usage.used_required_calls
                    if final_beacon_usage is not None else ()
                ),
                notes=(
                    final_beacon_usage.notes
                    if final_beacon_usage is not None else ()
                ),
            )
            self._print("memory write done")

        success = False
        if final_verifier is not None and final_verifier.ok:
            success = True
            if agent_cfg.require_beacon_usage_check:
                success = bool(final_beacon_usage is not None and final_beacon_usage.ok)

        self._print(f"workflow done success={success}")

        return AgentWorkflowResult(
            build=build,
            logic_acceptance=logic_acceptance,
            memory_read=memory_read,
            memory_prompt=memory_prompt,
            thoughts=tuple(thoughts),
            scores=tuple(scores),
            selected_thoughts=tuple(selected_thoughts),
            rounds=tuple(rounds),
            final_generation=final_generation,
            final_format_check=final_format_check,
            final_verifier=final_verifier,
            final_beacon_usage=final_beacon_usage,
            memory_write=memory_write,
            success=success,
            meta={
                "run_id": run_id,
                "task_id": task.id,
                "round_count": len(rounds),
                "selected_thought_id": selected_thought.id if selected_thought is not None else "",
            },
        )


def run_agent_workflow(
    *,
    llm: LLMClient,
    task: TaskObject,
    project_index: ProjectIndex,
    run_id: str = "",
    run_config: Optional[RunConfig] = None,
    reader_config: Optional[ReaderConfig] = None,
    agent_config: Optional[AgentConfig] = None,
    memory_store_path: str = "outputs/memory/experience.jsonl",
    print_io: bool = False,
) -> AgentWorkflowResult:
    """
    Convenience entry for direct workflow execution.
    """
    workflow = AgentWorkflow(
        llm=llm,
        memory_store_path=memory_store_path,
        print_io=print_io,
    )
    return workflow.run(
        task=task,
        project_index=project_index,
        run_id=run_id,
        run_config=run_config,
        reader_config=reader_config,
        agent_config=agent_config,
    )