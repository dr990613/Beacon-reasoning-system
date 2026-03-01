# src/beacon_system/logic/state.py
# -*- coding: utf-8 -*-
"""
state.py

ReasoningState is the ONLY mutable container used by Beacon Logic rules.
Rules (local/global) must only read/write this state; they must not "return IR".

MVP goals:
- Centralize all intermediate structures (AST index, nodes/edges, symbols, callgraph).
- Record provenance for every introduced node/edge.
- Keep the data model lightweight but extensible.

NOTE:
- This module intentionally depends only on logic primitives (anchors) and stdlib.
- Do not import generator/verifier/adapters here (Single Source of Truth boundary).
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, NewType, Iterable
from dataclasses import dataclass, field
from typing import Any, Dict
from .anchors import Anchor, NodeID, Namespace


# ---- Keys / lightweight identifiers ----

FuncKey = NewType("FuncKey", str)
SymbolName = NewType("SymbolName", str)


# ---- Provenance ----

class RuleName(str, Enum):
    # Local rules
    L_OUT = "L-OUT"
    L_DEP = "L-DEP"
    L_VAL = "L-VAL"
    L_RED = "L-RED"
    # Global rules
    G_BASE = "G-BASE"
    G_CALL = "G-CALL"
    G_RET = "G-RET"
    G_GLOB = "G-GLOB"
    P_ENTRY = "P-ENTRY"


@dataclass(frozen=True, slots=True)
class ProvenanceStep:
    """
    Records how a node/edge was introduced.

    Fields:
    - rule: which Beacon rule introduced it
    - src: optional source NodeID (e.g., dependency expansion from a parent beacon)
    - note: optional short text for debugging
    """
    rule: RuleName
    src: Optional[NodeID] = None
    note: Optional[str] = None


# ---- Beacon graph primitives ----

class EdgeKind(str, Enum):
    DATA = "data"
    CONTROL = "control"
    CALL = "call"
    RET = "ret"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class BeaconNode:
    """
    A Beacon node represents a semantically relevant program element.

    - node_id: stable identity (NodeID)
    - anchor: location
    - kind: AST kind (e.g. "Return", "Assign", "Call")
    - code: optional snippet (engine can fill from source later)
    - meta: extensible dict (e.g., symbol names, callee, etc.)
    """
    node_id: NodeID
    anchor: Anchor
    kind: str
    code: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class BeaconEdge:
    """
    Typed edge between nodes.

    Hash/eq MUST be stable and not depend on unhashable meta.
    We treat (src, dst, kind) as the identity for deduplication in sets.
    """
    src: NodeID
    dst: NodeID
    kind: EdgeKind
    meta: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def key(self) -> tuple:
        return (str(self.kind.value if hasattr(self.kind, "value") else self.kind), str(self.src), str(self.dst))

# ---- Symbols / Callgraph ----

@dataclass
class SymbolTable:
    """
    Symbol table for the current reasoning scope.

    MVP fields:
    - imports: imported module or imported names
    - globals: global variables/constants referenced
    - attrs: attribute names referenced (e.g. obj.attr)
    - calls: called function names (best-effort string names)
    """
    imports: Set[str] = field(default_factory=set)
    globals: Set[str] = field(default_factory=set)
    attrs: Set[str] = field(default_factory=set)
    calls: Set[str] = field(default_factory=set)

    def merge(self, other: "SymbolTable") -> None:
        self.imports |= other.imports
        self.globals |= other.globals
        self.attrs |= other.attrs
        self.calls |= other.calls

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "imports": sorted(self.imports),
            "globals": sorted(self.globals),
            "attrs": sorted(self.attrs),
            "calls": sorted(self.calls),
        }


# ---- Reader config (logic-only switches) ----

@dataclass(frozen=True, slots=True)
class ReaderConfig:
    """
    Configuration for Beacon reasoning (logic engine).

    enable_global: whether to run global reasoning rules
    validation_filter: whether to run L-VAL filtering
    max_local_nodes: optional cap for L-RED (normalize/reduce can enforce)
    max_global_inline: optional cap for G-CALL inlining expansions
    """
    enable_global: bool = True
    validation_filter: bool = True
    max_local_nodes: Optional[int] = None
    max_global_inline: Optional[int] = None


# ---- AST index ----

@dataclass
class ASTIndex:
    """
    Minimal project AST index.

    MVP:
    - files: map file path -> ast.Module
    - functions: map FuncKey -> ast.FunctionDef | ast.AsyncFunctionDef
    - source_lines: optional map file path -> list[str] (engine may inject)
    """
    files: Dict[str, ast.Module] = field(default_factory=dict)
    functions: Dict[FuncKey, ast.AST] = field(default_factory=dict)
    source_lines: Dict[str, List[str]] = field(default_factory=dict)

    def get_function(self, key: FuncKey) -> Optional[ast.AST]:
        return self.functions.get(key)

    def add_file(self, file: str, module_ast: ast.Module, source: Optional[str] = None) -> None:
        self.files[file] = module_ast
        if source is not None:
            self.source_lines[file] = source.splitlines()

    def register_function(self, file: str, qualname: str, fn_node: ast.AST) -> FuncKey:
        key = FuncKey(f"{file}::{qualname}")
        self.functions[key] = fn_node
        return key


# ---- Task placeholder typing ----
# We keep TaskObject opaque here to avoid circular dependencies.
# In your repo, import the real TaskObject from beacon_system.types once it exists.
TaskObject = Any


# ---- Reasoning State ----

@dataclass
class ReasoningState:
    """
    The single mutable state for Beacon Logic reasoning.

    Rules MUST:
    - only read/write ReasoningState
    - use add_node/add_edge helpers to ensure provenance is tracked
    """
    task: TaskObject
    ast_index: ASTIndex
    config: ReaderConfig = field(default_factory=ReaderConfig)

    # Beacon graph
    nodes: Dict[NodeID, BeaconNode] = field(default_factory=dict)
    edges: Set[BeaconEdge] = field(default_factory=set)

    # Analysis artifacts
    symbols: SymbolTable = field(default_factory=SymbolTable)
    callgraph: Dict[FuncKey, Set[FuncKey]] = field(default_factory=dict)

    # Rule bookkeeping
    provenance: Dict[NodeID, List[ProvenanceStep]] = field(default_factory=dict)
    forbidden: Set[NodeID] = field(default_factory=set)

    # Deterministic disambiguation counter per (file, qualname, lineno, col, kind)
    _local_counters: Dict[Tuple[str, str, int, int, str], int] = field(default_factory=dict, repr=False)

    def next_local_index(self, anchor: Anchor, kind: str) -> int:
        """
        Return a deterministic local_index for nodes that share the same anchor+kind.
        Rules can use this to construct NodeIDs consistently within a build.
        """
        key = (anchor.file, anchor.qualname, anchor.lineno, anchor.col, kind)
        idx = self._local_counters.get(key, 0)
        self._local_counters[key] = idx + 1
        return idx

    def add_node(
        self,
        node: BeaconNode,
        prov: ProvenanceStep,
        *,
        overwrite: bool = False,
    ) -> None:
        """
        Add a node and provenance. If node_id already exists:
        - overwrite=False: keep existing node but append provenance
        - overwrite=True: replace node (rare; prefer not to)
        """
        if node.node_id not in self.nodes or overwrite:
            self.nodes[node.node_id] = node
        self.provenance.setdefault(node.node_id, []).append(prov)

    def add_edge(self, edge: BeaconEdge, prov: ProvenanceStep) -> None:
        """
        Add an edge. Edges are stored as a set so they must be hashable (frozen dataclass).
        Provenance for edges is currently tracked on dst node (MVP) or via meta.
        If you want per-edge provenance later, add a separate edge_provenance map.
        """
        self.edges.add(edge)
        # MVP: attach edge provenance to dst node (conservative).
        self.provenance.setdefault(edge.dst, []).append(
            ProvenanceStep(rule=prov.rule, src=edge.src, note=prov.note or f"edge:{edge.kind.value}")
        )

    def mark_forbidden(self, node_id: NodeID, prov: Optional[ProvenanceStep] = None) -> None:
        self.forbidden.add(node_id)
        if prov is not None:
            self.provenance.setdefault(node_id, []).append(prov)

    def add_callgraph_edge(self, caller: FuncKey, callee: FuncKey) -> None:
        self.callgraph.setdefault(caller, set()).add(callee)

    # ---- Convenience getters ----

    def is_forbidden(self, node_id: NodeID) -> bool:
        return node_id in self.forbidden

    def get_node(self, node_id: NodeID) -> Optional[BeaconNode]:
        return self.nodes.get(node_id)

    def iter_nodes(self) -> Iterable[BeaconNode]:
        return self.nodes.values()

    def iter_edges(self) -> Iterable[BeaconEdge]:
        return self.edges