# -*- coding: utf-8 -*-

"""
Minimal Agent Workflow

Responsibilities:
- Orchestrate the minimal agent chain:
    1. logic build
    2. generate
    3. rebuild
    4. verify
    5. optional one-step revise
- No memory
- No planning
- No scoring
- One main generation + at most one revision

Stable external contracts:
- logic_engine.build(task, project_index, run_config) -> LogicBuildResult
- CodeGeneratorAgent.run(...)
- RebuilderAgent.run(...)
- BeaconVerifierAgent.run(...)

Design goals:
- Minimal
- Explicit
- Easy to debug
- Schema tolerant
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional


from .generator import CodeGeneratorAgent, GeneratorResult
from .rebuilder import RebuilderAgent, RebuildResult
from .verifier import BeaconVerifierAgent, VerifierResult
from ..llm.client import LLMClient


# ============================================================
# helpers
# ============================================================

def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        try:
            return dict(vars(obj))
        except Exception:
            pass
    return {}


def _pick_logic_field(logic_result: Any, key: str, default: Any = None) -> Any:
    if isinstance(logic_result, dict):
        return logic_result.get(key, default)
    return getattr(logic_result, key, default)


def _extract_logic_inputs(logic_result: Any) -> Dict[str, Any]:
    return {
        "beacon_tree": _pick_logic_field(logic_result, "beacon_tree"),
        "signature_hints": _pick_logic_field(logic_result, "signature_hints"),
        "constraint_summary": _pick_logic_field(logic_result, "constraint_summary"),
    }


def _build_revision_constraint_summary(
    original_constraint_summary: Any,
    ver_result: VerifierResult,
) -> Dict[str, Any]:
    """
    Inject verifier feedback into generator-facing constraints for one-step revise.
    Keep it simple and explicit.
    """
    base = _as_dict(original_constraint_summary)
    revised = dict(base)

    revised["verifier_revision_advice"] = ver_result.revision_advice
    revised["verifier_issues"] = ver_result.issues
    revised["revision_mode"] = "one_step_fix"

    return revised


# ============================================================
# result objects
# ============================================================

@dataclass
class WorkflowRoundResult:
    generation: GeneratorResult
    rebuild: RebuildResult
    verification: VerifierResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "generation": self.generation.to_dict(),
            "rebuild": self.rebuild.to_dict(),
            "verification": self.verification.to_dict(),
        }


@dataclass
class WorkflowResult:
    accepted: bool
    final_code: str
    logic_result: Any
    main_round: WorkflowRoundResult
    revise_round: Optional[WorkflowRoundResult]
    total_rounds: int
    stopped_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "final_code": self.final_code,
            "logic_result": _as_dict(self.logic_result) if not isinstance(self.logic_result, dict) else self.logic_result,
            "main_round": self.main_round.to_dict(),
            "revise_round": None if self.revise_round is None else self.revise_round.to_dict(),
            "total_rounds": self.total_rounds,
            "stopped_reason": self.stopped_reason,
        }


# ============================================================
# workflow
# ============================================================

class AgentWorkflow:
    """
    Minimal workflow:
        logic build -> generate -> rebuild -> verify -> optional revise once
    """

    def __init__(
        self,
        *,
        logic_engine: Any,
        llm_client: LLMClient,
        allow_one_step_revise: bool = True,
    ) -> None:
        if not hasattr(logic_engine, "build"):
            raise TypeError("logic_engine must expose build(task, project_index, run_config).")

        self.logic_engine = logic_engine
        self.llm_client = llm_client
        self.allow_one_step_revise = allow_one_step_revise

        self.generator = CodeGeneratorAgent(llm_client)
        self.rebuilder = RebuilderAgent(logic_engine=logic_engine)
        self.verifier = BeaconVerifierAgent(llm_client)

    def run(
        self,
        *,
        task: Any,
        project_index: Any,
        run_config: Any,
    ) -> WorkflowResult:
        """
        Execute the minimal agent chain.

        Flow:
        1. logic build
        2. main generate
        3. main rebuild
        4. main verify
        5. optional one-step revise
        """
        logic_result = self.logic_engine.build(task, project_index, run_config)

        logic_inputs = _extract_logic_inputs(logic_result)
        beacon_tree = logic_inputs["beacon_tree"]
        signature_hints = logic_inputs["signature_hints"]
        constraint_summary = logic_inputs["constraint_summary"]

        # -------------------------
        # main round
        # -------------------------
        gen_result = self.generator.run(
            task=task,
            beacon_tree=beacon_tree,
            signature_hints=signature_hints,
            constraint_summary=constraint_summary,
        )

        rebuild_result = self.rebuilder.run(
            task=task,
            generated_code=gen_result.generated_code,
            project_index=project_index,
            run_config=run_config,
        )

        ver_result = self.verifier.run(
            task=task,
            original_beacon=logic_result,
            rebuilt_beacon=rebuild_result.rebuilt_beacon,
            generated_code=gen_result.generated_code,
        )

        main_round = WorkflowRoundResult(
            generation=gen_result,
            rebuild=rebuild_result,
            verification=ver_result,
        )

        if ver_result.accepted:
            return WorkflowResult(
                accepted=True,
                final_code=gen_result.generated_code,
                logic_result=logic_result,
                main_round=main_round,
                revise_round=None,
                total_rounds=1,
                stopped_reason="accepted_after_main_round",
            )

        # -------------------------
        # optional one-step revise
        # -------------------------
        if not self.allow_one_step_revise:
            return WorkflowResult(
                accepted=False,
                final_code=gen_result.generated_code,
                logic_result=logic_result,
                main_round=main_round,
                revise_round=None,
                total_rounds=1,
                stopped_reason="rejected_after_main_round_no_revise",
            )

        revised_constraint_summary = _build_revision_constraint_summary(
            constraint_summary,
            ver_result,
        )

        revise_gen_result = self.generator.run(
            task=task,
            beacon_tree=beacon_tree,
            signature_hints=signature_hints,
            constraint_summary=revised_constraint_summary,
        )

        revise_rebuild_result = self.rebuilder.run(
            task=task,
            generated_code=revise_gen_result.generated_code,
            project_index=project_index,
            run_config=run_config,
        )

        revise_ver_result = self.verifier.run(
            task=task,
            original_beacon=logic_result,
            rebuilt_beacon=revise_rebuild_result.rebuilt_beacon,
            generated_code=revise_gen_result.generated_code,
        )

        revise_round = WorkflowRoundResult(
            generation=revise_gen_result,
            rebuild=revise_rebuild_result,
            verification=revise_ver_result,
        )

        if revise_ver_result.accepted:
            return WorkflowResult(
                accepted=True,
                final_code=revise_gen_result.generated_code,
                logic_result=logic_result,
                main_round=main_round,
                revise_round=revise_round,
                total_rounds=2,
                stopped_reason="accepted_after_revise_round",
            )

        return WorkflowResult(
            accepted=False,
            final_code=revise_gen_result.generated_code,
            logic_result=logic_result,
            main_round=main_round,
            revise_round=revise_round,
            total_rounds=2,
            stopped_reason="rejected_after_revise_round",
        )