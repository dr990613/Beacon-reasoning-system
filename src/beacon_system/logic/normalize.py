# src/beacon_system/logic/normalize.py
# -*- coding: utf-8 -*-
"""
normalize.py

Deterministic canonicalization for Beacon Logic artifacts.

Why this exists:
- Python sets/dicts are not stable across runs in a way we can rely on for research artifacts.
- We need byte-stable IR/Constraints outputs for reproducibility, ablations, debugging, and caching.

MVP responsibilities:
- canonicalize_nodes: dict[NodeID, BeaconNode] -> list[BeaconNode] (sorted by NodeID)
- canonicalize_edges: set[BeaconEdge] -> list[BeaconEdge] (sorted by (kind, src, dst))
- canonicalize_symbols: SymbolTable -> SymbolTable (sorted lists or stable dict)
- reduce_ir: remove hanging edges, optionally cap node count deterministically
- stable_json: stable JSON string (sorted keys, stable floats)

NOTE:
- This module should NOT implement Beacon reasoning rules; it only canonicalizes/normalizes outputs.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple

from .anchors import NodeID
from .state import BeaconEdge, BeaconNode, EdgeKind, SymbolTable, ReasoningState


# ----------------------------
# Stable JSON
# ----------------------------

def _stable_float(x: float) -> float:
    """
    Normalize float values to avoid platform-dependent representations.
    Keeps NaN/Inf in a deterministic way (as strings) during encoding.
    """
    if math.isnan(x):
        # JSON has no NaN; represent deterministically
        return "NaN"  # type: ignore[return-value]
    if math.isinf(x):
        return "Infinity" if x > 0 else "-Infinity"  # type: ignore[return-value]
    # Round to a conservative precision to stabilize representation.
    # If you need exactness later, raise precision.
    return float(f"{x:.12g}")


def _normalize_obj(obj: Any) -> Any:
    """
    Recursively normalize objects into JSON-serializable primitives with stable ordering.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        return _stable_float(obj)

    # Typed IDs
    if type(obj) is str:
        return obj

    # dataclasses
    if is_dataclass(obj):
        return _normalize_obj(asdict(obj))

    # dict-like
    if isinstance(obj, dict):
        # Sort keys deterministically
        items = sorted(obj.items(), key=lambda kv: str(kv[0]))
        return {str(k): _normalize_obj(v) for k, v in items}

    # set/tuple/list
    if isinstance(obj, (set, tuple, list)):
        # Convert to list and sort if elements look comparable
        lst = [_normalize_obj(x) for x in obj]
        # Try stable sort; if unorderable, keep insertion (already deterministic from our callers)
        try:
            return sorted(lst, key=lambda x: json.dumps(x, sort_keys=True, ensure_ascii=False))
        except Exception:
            return lst

    # Fallback to string
    return str(obj)


def stable_json(obj: Any) -> str:
    """
    Produce a stable JSON string:
    - sort_keys=True
    - ensure_ascii=False for readability
    - separators to avoid whitespace variance
    - stable float formatting via _normalize_obj

    Intended for determinism tests and hashing.
    """
    normalized = _normalize_obj(obj)
    return json.dumps(
        normalized,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )


# ----------------------------
# Canonicalization helpers
# ----------------------------

def canonicalize_nodes(nodes: Dict[NodeID, BeaconNode]) -> List[BeaconNode]:
    """
    Deterministically order nodes by NodeID (string order).
    """
    return [nodes[k] for k in sorted(nodes.keys(), key=str)]


def canonicalize_edges(edges: Set[BeaconEdge]) -> List[BeaconEdge]:
    """
    Deterministically order edges by (kind, src, dst), then stable-json of meta.
    """
    def edge_key(e: BeaconEdge) -> Tuple[str, str, str, str]:
        return (
            e.kind.value if isinstance(e.kind, EdgeKind) else str(e.kind),
            str(e.src),
            str(e.dst),
            stable_json(e.meta),
        )

    return sorted(list(edges), key=edge_key)


def canonicalize_symbols(symbols: SymbolTable) -> SymbolTable:
    """
    Ensure sets are deduped and stable. SymbolTable already stores sets;
    we return a fresh SymbolTable to avoid accidental mutation across stages.
    """
    return SymbolTable(
        imports=set(sorted(symbols.imports)),
        globals=set(sorted(symbols.globals)),
        attrs=set(sorted(symbols.attrs)),
        calls=set(sorted(symbols.calls)),
    )


# ----------------------------
# IR Reduction
# ----------------------------

def _prune_hanging_edges(nodes: Dict[NodeID, BeaconNode], edges: Set[BeaconEdge]) -> Set[BeaconEdge]:
    """
    Remove edges that reference missing nodes.
    """
    node_ids = set(nodes.keys())
    return {e for e in edges if (e.src in node_ids and e.dst in node_ids)}


def _deterministic_topk_node_ids(nodes: Dict[NodeID, BeaconNode], k: int) -> Set[NodeID]:
    """
    Deterministically select top-k NodeIDs by lexical order.
    This is intentionally simple for MVP.
    """
    if k <= 0:
        return set()
    return set(sorted(nodes.keys(), key=str)[:k])


def reduce_ir(state: ReasoningState) -> Tuple[Dict[NodeID, BeaconNode], Set[BeaconEdge], SymbolTable, Set[NodeID]]:
    """
    Reduce IR deterministically.

    MVP reduction:
    1) Prune edges referencing missing nodes.
    2) If config.max_local_nodes is set, cap the number of nodes deterministically.
       (In MVP we treat it as a global cap over state.nodes; later you can separate local/global caps.)
    3) Re-prune edges after capping.
    4) Ensure forbidden is subset of nodes (drop forbidden ids that no longer exist).
    """
    nodes = dict(state.nodes)
    edges = set(state.edges)
    symbols = canonicalize_symbols(state.symbols)
    forbidden = set(state.forbidden)

    # Step 1: prune edges that reference missing nodes
    edges = _prune_hanging_edges(nodes, edges)

    # Step 2: optional deterministic cap
    k = state.config.max_local_nodes
    if isinstance(k, int) and k > 0 and len(nodes) > k:
        keep_ids = _deterministic_topk_node_ids(nodes, k)
        nodes = {nid: n for nid, n in nodes.items() if nid in keep_ids}
        # Step 3: re-prune edges
        edges = _prune_hanging_edges(nodes, edges)
        # Step 4: forbidden subset
        forbidden = {nid for nid in forbidden if nid in nodes}

    # Final: ensure forbidden ids exist
    forbidden = {nid for nid in forbidden if nid in nodes}

    return nodes, edges, symbols, forbidden


# ----------------------------
# Convenience: canonical IR bundle
# ----------------------------

def canonical_ir_bundle(
    nodes: Dict[NodeID, BeaconNode],
    edges: Set[BeaconEdge],
    symbols: SymbolTable,
    forbidden: Set[NodeID],
) -> Dict[str, Any]:
    """
    Helpful for stable_json hashing / determinism tests.
    """
    return {
        "nodes": [asdict(n) for n in canonicalize_nodes(nodes)],
        "edges": [asdict(e) for e in canonicalize_edges(edges)],
        "symbols": symbols.to_dict(),
        "forbidden": sorted([str(x) for x in forbidden]),
    }