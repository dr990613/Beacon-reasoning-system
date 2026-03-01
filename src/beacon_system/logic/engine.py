# src/beacon_system/logic/engine.py
# -*- coding: utf-8 -*-
"""
engine.py

Beacon Logic Engine (Single Source of Truth) + Dual Outputs contract.

build() pipeline (MVP):
1) init ReasoningState (task + ASTIndex + config)
2) local reasoning on entry function
3) optional global reasoning
4) normalize/reduce (deterministic)
5) build BeaconIR (structured IR)
6) compile Constraints from BeaconIR
7) return BuildResult(ir, constraints, debug)

IMPORTANT:
- This module is the only orchestrator of logic rules.
- Verifier consumes Constraints only. Generator consumes BeaconIR + memory (later).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .state import ASTIndex, FuncKey, ReaderConfig, ReasoningState
from .rules_local import apply_local
from .rules_global import apply_global
from .normalize import reduce_ir, canonical_ir_bundle, stable_json
from .constraints import Constraints, compile_constraints


# Keep TaskObject/BeaconIR opaque for MVP, but we create a concrete BeaconIR dataclass here.
# Later you can move BeaconIR to beacon_system/types.py and import it.
TaskObject = Any


@dataclass(frozen=True, slots=True)
class BeaconIR:
    """
    Minimal Beacon IR (MVP).

    Fields:
    - nodes: list of beacon nodes (wire dicts or dataclasses)
    - edges: list of beacon edges
    - symbols: symbol dict (imports/globals/attrs/calls)
    - forbidden: list of forbidden node ids
    - skeleton: optional string scaffold (unused in MVP)
    - provenance: optional dict node_id -> provenance steps (for explainability)
    """
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    symbols: Dict[str, List[str]]
    forbidden: List[str]
    skeleton: Optional[str] = None
    provenance: Optional[Dict[str, Any]] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DebugBundle:
    """
    Optional debug bundle for research / inspection.
    Keep it light to avoid bloating artifacts.
    """
    state_stats: Dict[str, Any]
    ir_hash: str
    constraints_hash: str


@dataclass(frozen=True, slots=True)
class BuildResult:
    ir: BeaconIR
    constraints: Constraints
    debug: Optional[DebugBundle] = None


# ----------------------------
# ProjectIndex (MVP)
# ----------------------------

@dataclass
class ProjectIndex:
    """
    MVP ProjectIndex:
    - entry_file: path to file that contains entry function
    - entry_qualname: qualified name for entry function (e.g. "entry" or "Class.method")
    - files: map file path -> source code string

    Engine will parse these sources into ASTIndex.
    """
    entry_file: str
    entry_qualname: str
    files: Dict[str, str]


# ----------------------------
# Helpers
# ----------------------------

def _build_ast_index(project_index: ProjectIndex) -> ASTIndex:
    ai = ASTIndex()
    for file, source in project_index.files.items():
        mod = ast.parse(source)
        ai.add_file(file, mod, source=source)
        # Register all functions in this file (top-level and class methods)
        for fn_key, fn_node in _extract_functions(file, mod):
            ai.functions[fn_key] = fn_node
    return ai


def _extract_functions(file: str, module_ast: ast.Module) -> List[Tuple[FuncKey, ast.AST]]:
    """
    MVP function extractor:
    - top-level functions: qualname = fn.name
    - class methods: qualname = "ClassName.method"
    Nested functions are ignored.
    """
    out: List[Tuple[FuncKey, ast.AST]] = []

    for node in module_ast.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qual = node.name
            out.append((FuncKey(f"{file}::{qual}"), node))
        elif isinstance(node, ast.ClassDef):
            cls = node.name
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qual = f"{cls}.{sub.name}"
                    out.append((FuncKey(f"{file}::{qual}"), sub))
    return out


def _find_entry_key(ast_index: ASTIndex, entry_file: str, entry_qualname: str) -> Optional[FuncKey]:
    target = FuncKey(f"{entry_file}::{entry_qualname}")
    if target in ast_index.functions:
        return target
    # fallback: try suffix match
    for fk in ast_index.functions.keys():
        if str(fk).endswith(f"::{entry_qualname}"):
            return fk
    return None


def _node_to_wire(n: Any) -> Dict[str, Any]:
    # BeaconNode is a frozen dataclass; use its fields (avoid importing dataclasses.asdict for speed)
    return {
        "node_id": str(n.node_id),
        "anchor": n.anchor.to_dict(),
        "kind": n.kind,
        "code": n.code,
        "meta": dict(n.meta) if isinstance(getattr(n, "meta", None), dict) else {},
    }


def _edge_to_wire(e: Any) -> Dict[str, Any]:
    return {
        "src": str(e.src),
        "dst": str(e.dst),
        "kind": e.kind.value if hasattr(e.kind, "value") else str(e.kind),
        "meta": dict(e.meta) if isinstance(getattr(e, "meta", None), dict) else {},
    }


# ----------------------------
# Public API
# ----------------------------

def build(
    task: TaskObject,
    project_index: ProjectIndex,
    config: ReaderConfig,
    seed: int = 0,
    *,
    with_debug: bool = True,
) -> BuildResult:
    """
    Build BeaconIR and Constraints for a given task and project context.

    Determinism:
    - seed is accepted for future use; MVP build path is deterministic without randomness.
    - normalize.reduce_ir + stable_json ensure byte-stable artifacts for the same inputs.

    task:
    - MVP does not require a specific TaskObject schema, but expects (optionally):
      task.id / task.sig / task.doc / task.ctx etc.
    """
    _ = seed  # reserved for future

    # 1) Init ASTIndex
    ast_index = _build_ast_index(project_index)

    # 2) Locate entry function
    entry_key = _find_entry_key(ast_index, project_index.entry_file, project_index.entry_qualname)
    if entry_key is None:
        # Return empty but deterministic outputs
        empty_ir = BeaconIR(nodes=[], edges=[], symbols={"imports": [], "globals": [], "attrs": [], "calls": []}, forbidden=[])
        empty_constraints = compile_constraints(empty_ir, config)
        dbg = DebugBundle(state_stats={"error": "entry_not_found"}, ir_hash=_hash(stable_json(empty_ir)), constraints_hash=_hash(stable_json(empty_constraints.to_dict()))) if with_debug else None
        return BuildResult(ir=empty_ir, constraints=empty_constraints, debug=dbg)

    # 3) Init state
    state = ReasoningState(task=task, ast_index=ast_index, config=config)

    # 4) Local reasoning on entry
    apply_local(state, entry_key)

    # 5) Optional global reasoning
    if config.enable_global:
        apply_global(state, entry_key)

    # 6) Normalize / reduce deterministically
    nodes_d, edges_s, symbols, forbidden_s = reduce_ir(state)

    # 7) Build BeaconIR (wire form)
    nodes_wire = [_node_to_wire(n) for _, n in sorted(nodes_d.items(), key=lambda kv: str(kv[0]))]
    edges_wire = [_edge_to_wire(e) for e in sorted(list(edges_s), key=lambda e: (e.kind.value if hasattr(e.kind, "value") else str(e.kind), str(e.src), str(e.dst)))]

    # provenance wire (optional, stable)
    prov_wire = {
        str(nid): [{"rule": p.rule.value, "src": str(p.src) if p.src else None, "note": p.note} for p in state.provenance.get(nid, [])]
        for nid in sorted(nodes_d.keys(), key=str)
    }

    ir = BeaconIR(
        nodes=nodes_wire,
        edges=edges_wire,
        symbols=symbols.to_dict(),
        forbidden=sorted([str(x) for x in forbidden_s]),
        skeleton=None,
        provenance=prov_wire,
        meta={
            "entry_key": str(entry_key),
            "entry_file": project_index.entry_file,
            "entry_qualname": project_index.entry_qualname,
            "schema_version": "mvp-0.1",
        },
    )

    # 8) Compile Constraints from IR
    constraints = compile_constraints(ir, config)

    # 9) Optional debug bundle
    dbg = None
    if with_debug:
        ir_hash = _hash(stable_json(ir))
        cons_hash = _hash(stable_json(constraints.to_dict()))
        dbg = DebugBundle(
            state_stats={
                "n_nodes": len(nodes_d),
                "n_edges": len(edges_s),
                "n_forbidden": len(forbidden_s),
                "symbols": symbols.to_dict(),
            },
            ir_hash=ir_hash,
            constraints_hash=cons_hash,
        )

    return BuildResult(ir=ir, constraints=constraints, debug=dbg)


def _hash(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()