# src/beacon_system/logic/constraints.py
# -*- coding: utf-8 -*-
"""
constraints.py

Compile BeaconIR -> Constraints (the ONLY verifier input).

Contract:
- logic.engine.build() MUST output both BeaconIR and Constraints.
- Verifier MUST consume Constraints only (match_specs + required/forbidden), and MUST NOT re-run reasoning.

MVP Constraints fields:
- required_symbols: list[str]
- required_calls: list[str]
- forbidden_specs: list[MatchSpec]
- match_specs: list[MatchSpec]
- meta: {source_ir_hash, build_flags}

Compilation strategy (MVP):
- required_symbols: from ir.symbols (imports/globals/attrs/calls), flattened
- required_calls: union of:
  (a) ir.edges(kind=call).meta.call_name
  (b) ir.symbols.calls
  (c) ir.nodes[].meta.assign_from_call.call_name   <-- NEW fallback to avoid missing calls
- forbidden_specs: from ir.forbidden (best-effort; MVP uses ForbidPattern heuristics)
- match_specs: from ir.skeleton if present (MVP: empty)

NOTE:
- This module depends on matchers.py primitives; it does not depend on verifier.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Set, Tuple

from .matchers import (
    MatchSpec,
    HasCall,
    HasImport,
    HasName,
    ForbidCall,
    ForbidPattern,
    CallChain,
    AssignFromCall,
)
from .normalize import stable_json
from .state import EdgeKind, ReaderConfig


# Keep BeaconIR opaque here to avoid tight coupling; engine/types will pass the real BeaconIR.
BeaconIR = Any


@dataclass(frozen=True, slots=True)
class Constraints:
    """
    The only input for Verifier.

    required_symbols/calls are "soft-hard" requirements (presence checks).
    forbidden_specs/match_specs are executable MatchSpec rules.
    """
    required_symbols: Tuple[str, ...] = ()
    required_calls: Tuple[str, ...] = ()
    forbidden_specs: Tuple[MatchSpec, ...] = ()
    match_specs: Tuple[MatchSpec, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        # MatchSpec dataclasses are serializable via __dict__ or dataclasses.asdict,
        # but to keep it deterministic we store as stable_json strings per spec.
        return {
            "required_symbols": list(self.required_symbols),
            "required_calls": list(self.required_calls),
            "forbidden_specs": [stable_json(_spec_to_wire(s)) for s in self.forbidden_specs],
            "match_specs": [stable_json(_spec_to_wire(s)) for s in self.match_specs],
            "meta": dict(self.meta),
        }


def _spec_to_wire(spec: MatchSpec) -> Dict[str, Any]:
    """
    Convert MatchSpec into a stable wire dict (type + fields).
    """
    d = {"type": type(spec).__name__}
    # dataclasses with slots still support attribute access
    d.update({k: getattr(spec, k) for k in dir(spec) if not k.startswith("_") and k not in ("__class__",)})

    # The above may include methods; filter to json-serializable primitives
    clean: Dict[str, Any] = {"type": type(spec).__name__}
    for k, v in d.items():
        if k == "type":
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            clean[k] = v
        elif isinstance(v, (tuple, list)):
            clean[k] = list(v)
        elif isinstance(v, dict):
            clean[k] = v
    return clean


def _extract_symbols(ir: BeaconIR) -> Set[str]:
    """
    Flatten IR symbols. We accept several shapes:
    - ir.symbols is a dict with keys: imports/globals/attrs/calls
    - ir.symbols is an object with attributes .imports/.globals/.attrs/.calls
    """
    out: Set[str] = set()
    syms = getattr(ir, "symbols", None)
    if syms is None:
        return out

    if isinstance(syms, dict):
        for k in ("imports", "globals", "attrs", "calls"):
            vals = syms.get(k, [])
            if isinstance(vals, (set, list, tuple)):
                out |= {str(x) for x in vals}
    else:
        for k in ("imports", "globals", "attrs", "calls"):
            vals = getattr(syms, k, None)
            if isinstance(vals, (set, list, tuple)):
                out |= {str(x) for x in vals}

    out.discard("")
    return out


def _extract_calls_from_edges(ir: BeaconIR) -> Set[str]:
    """
    Extract required call names from IR call edges.
    Accept shapes:
    - ir.edges is list of dicts or list of objects with fields: kind/meta
    """
    calls: Set[str] = set()
    edges = getattr(ir, "edges", None)
    if edges is None:
        return calls

    for e in edges:
        kind = None
        meta = None
        if isinstance(e, dict):
            kind = e.get("kind")
            meta = e.get("meta", {}) or {}
        else:
            kind = getattr(e, "kind", None)
            meta = getattr(e, "meta", {}) or {}

        kind_s = kind.value if hasattr(kind, "value") else str(kind)
        if kind_s != EdgeKind.CALL.value:
            continue

        if isinstance(meta, dict):
            cn = meta.get("call_name") or meta.get("func") or meta.get("callee_name")
            if cn:
                calls.add(str(cn))

    calls.discard("")
    return calls


def _extract_calls_from_symbols(ir: BeaconIR) -> Set[str]:
    """
    Best-effort: collect call names from symbol table (ir.symbols.calls).
    """
    out: Set[str] = set()
    syms = getattr(ir, "symbols", None)
    calls = None
    if isinstance(syms, dict):
        calls = syms.get("calls")
    else:
        calls = getattr(syms, "calls", None)

    if isinstance(calls, (set, list, tuple)):
        out |= {str(x) for x in calls if str(x)}

    out.discard("")
    return out


def _extract_calls_from_assign_meta(ir: BeaconIR) -> Set[str]:
    """
    Fallback: collect call names from node.meta.assign_from_call.
    Works with wire-form BeaconIR nodes (dict):
      {"meta": {"assign_from_call": {"var": "...", "call_name": "compute"}}}
    """
    out: Set[str] = set()
    nodes = getattr(ir, "nodes", None)

    if not isinstance(nodes, list):
        return out

    for n in nodes:
        if not isinstance(n, dict):
            continue
        meta = n.get("meta") or {}
        if not isinstance(meta, dict):
            continue
        afc = meta.get("assign_from_call")
        if isinstance(afc, dict):
            cn = afc.get("call_name")
            if cn:
                out.add(str(cn))

    out.discard("")
    return out


def _extract_forbidden_ids(ir: BeaconIR) -> List[str]:
    """
    Extract forbidden NodeIDs from IR.
    Accept shapes:
    - ir.forbidden is iterable of NodeID/str
    - ir.forbidden_nodes, ir.forbidden_ids
    """
    for attr in ("forbidden", "forbidden_nodes", "forbidden_ids"):
        v = getattr(ir, attr, None)
        if v is None:
            continue
        if isinstance(v, (set, list, tuple)):
            return [str(x) for x in v]
    return []


def _make_forbidden_specs_mvp(ir: BeaconIR) -> List[MatchSpec]:
    """
    MVP forbidden compilation uses conservative heuristics for early-exit guards.
    """
    forbidden_ids = _extract_forbidden_ids(ir)
    if not forbidden_ids:
        return []

    patterns = [
        r"\bif\s+[^:\n]+\s*:\s*\n\s*return\b",
        r"\bif\s+[^:\n]+\s*:\s*\n\s*raise\b",
    ]
    return [ForbidPattern(regex=p) for p in patterns]


def _infer_match_specs_from_skeleton(ir: BeaconIR) -> List[MatchSpec]:
    """
    MVP: skeleton-to-spec compilation is optional.
    """
    _ = getattr(ir, "skeleton", None)
    return []


def compile_constraints(ir: BeaconIR, config: ReaderConfig) -> Constraints:
    """
    Compile BeaconIR into Constraints.

    Determinism:
    - All lists are sorted and deduped.
    - meta includes a stable hash of the source IR payload.
    """
    required_symbols = sorted(_extract_symbols(ir))

    # REQUIRED CALLS: union of 3 sources (edges, symbols, assign_from_call meta)
    calls_edges = _extract_calls_from_edges(ir)
    calls_symbols = _extract_calls_from_symbols(ir)
    calls_meta = _extract_calls_from_assign_meta(ir)
    required_calls = sorted((calls_edges | calls_symbols | calls_meta) - {""})

    forbidden_specs = _make_forbidden_specs_mvp(ir)
    match_specs = _infer_match_specs_from_skeleton(ir)

    # Build flags snapshot
    build_flags = {
        "enable_global": bool(getattr(config, "enable_global", True)),
        "validation_filter": bool(getattr(config, "validation_filter", True)),
        "max_local_nodes": getattr(config, "max_local_nodes", None),
        "max_global_inline": getattr(config, "max_global_inline", None),
    }

    # Stable IR hash: use stable_json over a limited set of IR fields if possible
    ir_wire = {
        "nodes": getattr(ir, "nodes", None),
        "edges": getattr(ir, "edges", None),
        "symbols": getattr(ir, "symbols", None),
        "forbidden": getattr(ir, "forbidden", getattr(ir, "forbidden_nodes", None)),
        "skeleton": getattr(ir, "skeleton", None),
    }
    source_ir_hash = _hash_str(stable_json(ir_wire))

    meta = {
        "source_ir_hash": source_ir_hash,
        "build_flags": build_flags,
        "constraints_version": "mvp-0.1",
    }

    return Constraints(
        required_symbols=tuple(required_symbols),
        required_calls=tuple(required_calls),
        forbidden_specs=tuple(forbidden_specs),
        match_specs=tuple(match_specs),
        meta=meta,
    )


def _hash_str(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8")).hexdigest()