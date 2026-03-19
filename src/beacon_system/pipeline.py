# src/beacon_system/pipeline.py
# -*- coding: utf-8 -*-

"""
Top-level pipeline orchestration.

Scope:
- Connect ONLY:
  adapter -> agent workflow -> runtime -> io
- Keep this as the single outer orchestrator.
- Do not place reasoning / planning / scoring / generation logic here.

Non-goals:
- no prompt logic
- no verifier logic
- no memory logic
- no runtime patch implementation details
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .adapters.base import RuntimeAdapter, TaskAdapter
from .agents.workflow import AgentWorkflow, AgentWorkflowResult
from .io import ensure_run_dir, make_run_id, write_artifacts
from .llm.client import LLMClient
from .types import (
    ExecResult,
    GenerationPayload,
    PipelineResult,
    RunConfig,
    TaskObject,
)


def _safe_code_from_generation(generation: Optional[GenerationPayload]) -> str:
    if generation is None or generation.primary is None:
        return ""
    return str(generation.primary.content or "")


def _build_patch(task: TaskObject, generation: Optional[GenerationPayload]) -> Dict[str, Any]:
    """
    Minimal runtime patch contract.

    Keep patch shape simple for adapter compatibility.
    """
    return {
        "target_file": str(task.target.get("file") or ""),
        "target_qualname": str(task.target.get("qualname") or ""),
        "new_code": _safe_code_from_generation(generation),
    }


@dataclass
class Pipeline:
    """
    Final outer pipeline orchestrator.

    Responsibilities:
    - build task/index via TaskAdapter
    - run AgentWorkflow
    - optionally run RuntimeAdapter on final generated code
    - persist artifacts via io.py
    """
    llm: LLMClient
    task_adapter: TaskAdapter
    runtime_adapter: RuntimeAdapter
    config: RunConfig
    memory_store_path: str = "outputs/memory/experience.jsonl"
    print_io: bool = False

    def _print(self, message: str) -> None:
        if self.print_io:
            print(f"[Pipeline] {message}")

    def run(self, *, run_id: Optional[str] = None) -> PipelineResult:
        run_id = str(run_id or make_run_id())
        run_dir = ensure_run_dir(self.config.outputs_dir, run_id)
        self._print(f"start run_id={run_id}")
        self._print(f"run_dir={run_dir}")

        # --------------------------------------------------
        # 1) adapter entry
        # --------------------------------------------------
        task, project_index = self.task_adapter.build_task()
        adapter_snapshot = self.task_adapter.snapshot()
        self._print(f"task built: task_id={task.id}")

        # --------------------------------------------------
        # 2) agent workflow
        # --------------------------------------------------
        workflow = AgentWorkflow(
            llm=self.llm,
            memory_store_path=self.memory_store_path,
            print_io=self.print_io,
        )
        workflow_result: AgentWorkflowResult = workflow.run(
            task=task,
            project_index=project_index,
            run_id=run_id,
            run_config=self.config,
        )
        self._print(f"workflow finished success={workflow_result.success}")

        # --------------------------------------------------
        # 3) persist agent-side rounds first
        # --------------------------------------------------
        for round_result in workflow_result.rounds:
            write_artifacts(
                run_dir,
                config=self.config,
                adapter_snapshot=adapter_snapshot,
                task=task,
                ir=workflow_result.build.ir,
                constraints=workflow_result.build.constraints,
                round_id=round_result.round_index,
                generation=round_result.generation,
                format_check=round_result.format_check,
                report=round_result.verifier,
                beacon_usage=round_result.beacon_usage,
                exec_result=None,
                thoughts=workflow_result.thoughts if round_result.round_index == 1 else None,
                scores=workflow_result.scores if round_result.round_index == 1 else None,
                memory_read=workflow_result.memory_read if round_result.round_index == 1 else None,
                memory_write=None,
            )

        # --------------------------------------------------
        # 4) runtime execution on final generation
        # --------------------------------------------------
        final_exec: Optional[ExecResult] = None
        final_generation = workflow_result.final_generation

        if final_generation is not None:
            patch = _build_patch(task, final_generation)
            self._print("runtime start")
            final_exec = self.runtime_adapter.run(task, patch)
            self._print(
                f"runtime finished status={final_exec.status} return_code={final_exec.return_code}"
            )

            final_round_id = len(workflow_result.rounds) if workflow_result.rounds else 1
            write_artifacts(
                run_dir,
                config=self.config,
                adapter_snapshot=adapter_snapshot,
                task=task,
                ir=workflow_result.build.ir,
                constraints=workflow_result.build.constraints,
                round_id=final_round_id,
                generation=final_generation,
                format_check=workflow_result.final_format_check,
                report=workflow_result.final_verifier,
                beacon_usage=workflow_result.final_beacon_usage,
                exec_result=final_exec,
                thoughts=None,
                scores=None,
                memory_read=None,
                memory_write=workflow_result.memory_write,
            )
        else:
            # still persist memory_write if no final generation exists
            final_round_id = len(workflow_result.rounds) if workflow_result.rounds else 1
            write_artifacts(
                run_dir,
                config=self.config,
                adapter_snapshot=adapter_snapshot,
                task=task,
                ir=workflow_result.build.ir,
                constraints=workflow_result.build.constraints,
                round_id=final_round_id,
                generation=None,
                format_check=workflow_result.final_format_check,
                report=workflow_result.final_verifier,
                beacon_usage=workflow_result.final_beacon_usage,
                exec_result=None,
                thoughts=None,
                scores=None,
                memory_read=None,
                memory_write=workflow_result.memory_write,
            )

        # --------------------------------------------------
        # 5) final success
        # --------------------------------------------------
        success = bool(workflow_result.success)
        if final_exec is not None:
            success = success and (final_exec.status == "pass")

        self._print(f"pipeline done success={success}")

        return PipelineResult(
            task=task,
            build=workflow_result.build,
            rounds=workflow_result.rounds,
            final_generation=workflow_result.final_generation,
            final_verifier=workflow_result.final_verifier,
            final_exec=final_exec,
            success=success,
            meta={
                "run_id": run_id,
                "run_dir": run_dir,
                "adapter_snapshot": adapter_snapshot,
                "workflow_meta": dict(workflow_result.meta or {}),
                "runtime_snapshot": self.runtime_adapter.snapshot(),
            },
        )


def run_pipeline(
    *,
    llm: LLMClient,
    task_adapter: TaskAdapter,
    runtime_adapter: RuntimeAdapter,
    config: RunConfig,
    run_id: Optional[str] = None,
    memory_store_path: str = "outputs/memory/experience.jsonl",
    print_io: bool = False,
) -> PipelineResult:
    """
    Convenience entry for direct pipeline execution.
    """
    pipeline = Pipeline(
        llm=llm,
        task_adapter=task_adapter,
        runtime_adapter=runtime_adapter,
        config=config,
        memory_store_path=memory_store_path,
        print_io=print_io,
    )
    return pipeline.run(run_id=run_id)