# -*- coding: utf-8 -*-

"""
Top-level pipeline orchestrator.

Responsibilities:
- load task/project inputs through adapter
- start logic/agent workflow
- persist artifacts
- return final structured pipeline result

Non-goals:
- no Beacon reasoning logic here
- no prompt construction here
- no patch logic here
- no verification logic here

Expected external contracts:
- task_adapter.load(raw_task=..., project_root=...) -> (TaskObject, ProjectIndex)
- logic_engine.build(task, project_index, run_config) -> LogicBuildResult
- AgentWorkflow.run(task=..., project_index=..., run_config=...) -> WorkflowResult
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from .agents.workflow import AgentWorkflow
from .io import (
    save_generation_artifacts,
    save_logic_artifacts,
    save_run_trace,
    save_verification_artifacts,
)


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "to_dict"):
        try:
            data = obj.to_dict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(vars(obj))
        except Exception:
            pass
    return {}


def _pick_task_id(task: Any) -> str:
    if isinstance(task, dict):
        value = task.get("task_id") or task.get("_id")
        return str(value) if value is not None else "unknown_task"

    for key in ("task_id", "_id"):
        value = getattr(task, key, None)
        if value is not None:
            return str(value)

    return "unknown_task"


def _resolve_output_dir(run_config: Any, default: str = "outputs") -> str:
    if not isinstance(run_config, dict):
        return default

    artifacts = run_config.get("artifacts", {})
    if isinstance(artifacts, dict):
        value = artifacts.get("output_dir")
        if isinstance(value, str) and value.strip():
            return value.strip()

    return default


def _extract_round_artifacts(round_result: Any) -> Dict[str, Any]:
    """
    Convert a WorkflowRoundResult-like object into plain dict sections.
    """
    if round_result is None:
        return {}

    if hasattr(round_result, "to_dict"):
        try:
            data = round_result.to_dict()
            if isinstance(data, dict):
                return data
        except Exception:
            pass

    return _as_dict(round_result)


@dataclass
class PipelineResult:
    task_id: str
    task: Any
    project_index: Any
    workflow_result: Any
    artifact_paths: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task": _as_dict(self.task),
            "project_index": _as_dict(self.project_index),
            "workflow_result": (
                self.workflow_result.to_dict()
                if hasattr(self.workflow_result, "to_dict")
                else _as_dict(self.workflow_result)
            ),
            "artifact_paths": self.artifact_paths,
        }


class BeaconPipeline:
    """
    Top-level orchestrator.

    Flow:
    1. adapter load raw_task + repo -> TaskObject + ProjectIndex
    2. start AgentWorkflow
    3. save artifacts
    4. return PipelineResult
    """

    def __init__(
        self,
        *,
        task_adapter: Any,
        logic_engine: Any,
        llm_client: Any,
    ) -> None:
        if not hasattr(task_adapter, "load"):
            raise TypeError("task_adapter must expose load(raw_task=..., project_root=...).")
        if not hasattr(logic_engine, "build"):
            raise TypeError("logic_engine must expose build(task, project_index, run_config).")

        self.task_adapter = task_adapter
        self.logic_engine = logic_engine
        self.llm_client = llm_client
        self.workflow = AgentWorkflow(
            logic_engine=logic_engine,
            llm_client=llm_client,
            allow_one_step_revise=True,
        )

    def run(
        self,
        *,
        raw_task: Dict[str, Any],
        project_root: str | Path,
        run_config: Any,
    ) -> PipelineResult:
        """
        Execute one full pipeline run.

        Required inputs:
        - raw_task: raw benchmark/task dict
        - project_root: repository root
        - run_config: runtime config dict
        """
        if raw_task is None or not isinstance(raw_task, dict):
            raise ValueError("raw_task must be a non-empty dict.")
        if project_root is None:
            raise ValueError("project_root must be provided.")

        task, project_index = self.task_adapter.load(
            raw_task=raw_task,
            project_root=project_root,
        )

        workflow_result = self.workflow.run(
            task=task,
            project_index=project_index,
            run_config=run_config,
        )

        task_id = _pick_task_id(task)
        output_dir = _resolve_output_dir(run_config, default="outputs")

        artifact_paths: Dict[str, Any] = {
            "logic": {},
            "main_round": {},
            "revise_round": {},
            "run_trace": None,
        }

        # -------------------------
        # logic artifacts
        # -------------------------
        logic_result = getattr(workflow_result, "logic_result", None)
        if logic_result is not None:
            artifact_paths["logic"] = save_logic_artifacts(
                output_dir=output_dir,
                task_id=task_id,
                logic_result=logic_result,
            )

        # -------------------------
        # main round artifacts
        # -------------------------
        main_round = getattr(workflow_result, "main_round", None)
        main_round_dict = _extract_round_artifacts(main_round)

        if isinstance(main_round_dict, dict):
            generation = main_round_dict.get("generation")
            verification = main_round_dict.get("verification")

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

        # -------------------------
        # revise round artifacts
        # -------------------------
        revise_round = getattr(workflow_result, "revise_round", None)
        revise_round_dict = _extract_round_artifacts(revise_round)

        if isinstance(revise_round_dict, dict) and revise_round_dict:
            generation = revise_round_dict.get("generation")
            verification = revise_round_dict.get("verification")

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

        # -------------------------
        # full run trace
        # -------------------------
        run_trace_payload = {
            "task_id": task_id,
            "task": _as_dict(task),
            "project_index": _as_dict(project_index),
            "workflow_result": (
                workflow_result.to_dict()
                if hasattr(workflow_result, "to_dict")
                else _as_dict(workflow_result)
            ),
        }

        artifact_paths["run_trace"] = save_run_trace(
            output_dir=output_dir,
            task_id=task_id,
            run_trace=run_trace_payload,
        )

        return PipelineResult(
            task_id=task_id,
            task=task,
            project_index=project_index,
            workflow_result=workflow_result,
            artifact_paths=artifact_paths,
        )