# src/beacon_system/logic/engine.py
# -*- coding: utf-8 -*-

"""
Logic Engine: single stable entry for Beacon Logic pipeline.

Pipeline:
    preprocess -> local -> global -> builder -> tree -> signatures -> refiner

Design goals:
- expose ONE stable public interface only
- keep output contract stable
- do not allow callers to assemble logic outputs manually
- keep implementation simple and deterministic where possible
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

from .preprocess import preprocess_task
from .rules_local import build_local_beacons
from .rules_global import build_global_beacons
from .builder import build_raw_ir
from .tree import build_beacon_tree
from .signatures import build_signature_hints
from .refiner import refine_beacon_tree


# ============================================================
# Stable output contract
# ============================================================

@dataclass
class LogicBuildResult:
    raw_ir: Dict[str, Any]
    beacon_tree: Dict[str, Any]
    signature_hints: Dict[str, Any]
    constraint_summary: Dict[str, Any]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Engine
# ============================================================

class LogicEngine:
    """
    Single orchestration entry for Beacon Logic.
    """

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        self.llm_client = llm_client

    def build(
        self,
        task: Any,
        project_index: Optional[Any] = None,
        run_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Stable public interface.

        Args:
            task:
                task object / dict-like input
            project_index:
                optional project context holder; currently passed through to debug
            run_config:
                runtime config dict

        Returns:
            LogicBuildResult as dict
        """
        cfg = run_config or {}

        # 1) preprocess
        preprocessed = preprocess_task(task)

        # 2) local
        local_result = build_local_beacons(preprocessed)

        # 3) global
        global_result = build_global_beacons(preprocessed, local_result)

        # 4) raw ir
        raw_ir = build_raw_ir(preprocessed, local_result, global_result)

        # 5) beacon tree
        beacon_tree = build_beacon_tree(raw_ir)

        # 6) signatures
        signature_hints = build_signature_hints(preprocessed, raw_ir)

        # 7) refiner
        refined_tree = refine_beacon_tree(
            beacon_tree=beacon_tree,
            signature_hints=signature_hints,
            llm_client=self.llm_client,
            run_config=cfg,
        )

        # 8) merge final visible beacon tree text
        final_beacon_tree = _merge_refined_tree(
            beacon_tree=beacon_tree,
            refined_tree=refined_tree,
        )

        # 9) minimal constraint summary
        constraint_summary = _build_constraint_summary(
            raw_ir=raw_ir,
            signature_hints=signature_hints,
        )

        result = LogicBuildResult(
            raw_ir=raw_ir,
            beacon_tree=final_beacon_tree,
            signature_hints=signature_hints,
            constraint_summary=constraint_summary,
            debug={
                "project_index_present": project_index is not None,
                "preprocess": {
                    "lang": preprocessed.get("lang"),
                    "warnings": preprocessed.get("warnings", []),
                    "target_name": preprocessed.get("target_name"),
                    "target_signature": preprocessed.get("target_signature"),
                },
                "local": {
                    "warnings": local_result.get("warnings", []),
                    "function_count": len(local_result.get("functions", []) or []),
                },
                "global": {
                    "warnings": global_result.get("warnings", []),
                    "entry_functions": global_result.get("entry_functions", []),
                    "program_beacon_node_count": len(global_result.get("program_beacon_node_ids", []) or []),
                },
                "builder": raw_ir.get("debug", {}),
                "tree": beacon_tree.get("debug", {}),
                "signatures": signature_hints.get("debug", {}),
                "refiner": {
                    "accepted": refined_tree.get("accepted", False),
                    "warnings": refined_tree.get("warnings", []),
                    "audit": refined_tree.get("audit", {}),
                },
            },
        )
        return result.to_dict()


# ============================================================
# Convenience function
# ============================================================

def build(
    task: Any,
    project_index: Optional[Any] = None,
    run_config: Optional[Dict[str, Any]] = None,
    llm_client: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Module-level stable interface.

    This is the only interface external callers should use.
    """
    engine = LogicEngine(llm_client=llm_client)
    return engine.build(
        task=task,
        project_index=project_index,
        run_config=run_config,
    )


# ============================================================
# Internal helpers
# ============================================================

def _merge_refined_tree(
    beacon_tree: Dict[str, Any],
    refined_tree: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Keep original tree structure, but replace visible rendered text
    with refined text when accepted.
    """
    out = dict(beacon_tree)

    refined_text = str(refined_tree.get("refined_text", "") or "").strip()
    accepted = bool(refined_tree.get("accepted", False))

    if accepted and refined_text:
        out["rendered_text"] = refined_text

    out["refiner_notes"] = list(refined_tree.get("notes", []) or [])
    out["refiner_audit"] = dict(refined_tree.get("audit", {}) or {})
    out["refiner_warnings"] = list(refined_tree.get("warnings", []) or [])
    return out


def _build_constraint_summary(
    raw_ir: Dict[str, Any],
    signature_hints: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Minimal constraint summary only.

    Important:
    - not a replacement for beacon_tree
    - not a compressed semantic core
    - only a lightweight helper for downstream generator/verifier
    """
    functions = raw_ir.get("functions", []) or []
    edges = raw_ir.get("edges", []) or []
    sig_functions = signature_hints.get("functions", []) or []

    required_functions = []
    for fn in functions:
        fn_name = fn.get("function_name")
        if fn_name:
            required_functions.append(fn_name)

    key_calls = []
    for sig_fn in sig_functions:
        for call in sig_fn.get("call_hints", []) or []:
            callee = call.get("callee")
            if callee:
                key_calls.append(callee)

    return {
        "required_functions": _unique_keep_order(required_functions),
        "key_calls": _unique_keep_order(key_calls),
        "edge_types_present": _unique_keep_order(
            [str(e.get("edge_type", "")) for e in edges if e.get("edge_type")]
        ),
    }


def _unique_keep_order(items):
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out