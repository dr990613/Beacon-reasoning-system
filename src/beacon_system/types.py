# src/beacon_system/types.py
# -*- coding: utf-8 -*-

"""
Contracts (single source of truth)

All cross-module data contracts live here to avoid drift.
- Use frozen dataclasses where possible to reduce accidental mutation.
- Prefer tuple[...] for stable ordering / deterministic serialization.
- All objects should be stable_json serializable via beacon_system.io.stable_json.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# ----------------------------
# Core task/index/config contracts
# ----------------------------


@dataclass(frozen=True)
class TaskTarget:
    file: str
    qualname: str


@dataclass(frozen=True)
class TaskObject:
    id: str
    lang: str
    level: str  # "function" | "file" | "project" (or enum)
    target: Dict[str, str]  # {"file": ..., "qualname": ...} keep dict to match contract text
    spec: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectIndex:
    """
    ProjectIndex is built by adapters and treated as read-only by logic.

    IMPORTANT (aligned with current logic.engine):
    - entry_file / entry_qualname are required for locating the entry function.
    - files is a mapping {relative_file_path: source_code_str} so logic can build ASTIndex deterministically.
    """
    root: str
    entry_file: str
    entry_qualname: str
    files: Dict[str, str]  # file -> source (OpenAI/benchmarks often provide sources; localrepo reads from disk)
    ast_index: Dict[str, Any] = field(default_factory=dict)  # file -> AstUnit (MVP: Any)
    symbols: Dict[str, Any] = field(default_factory=dict)    # optional coarse info
    callgraph: Dict[str, Any] = field(default_factory=dict)  # FuncKey -> set[FuncKey] (MVP: Any)


@dataclass(frozen=True)
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

# ----------------------------
# IR contracts
# ----------------------------

NodeID = str


@dataclass(frozen=True)
class Anchor:
    file: str
    qualname: str
    lineno: int
    col: int
    end_lineno: int
    end_col: int
    namespace: str  # LOCAL/GLOBAL/MODULE/CLASS/FUNCTION


@dataclass(frozen=True)
class ProvenanceStep:
    rule_id: str     # "L-OUT" | "L-DEP" | ...
    source: str      # source anchor or node id
    note: str = ""   # optional


@dataclass(frozen=True)
class BeaconNode:
    id: NodeID
    kind: str
    text: str
    anchor: Anchor
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BeaconEdge:
    kind: str         # "data" | "control" | "call"
    src: NodeID
    dst: NodeID
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Symbols:
    imports: Tuple[str, ...] = ()
    globals: Tuple[str, ...] = ()
    attrs: Tuple[str, ...] = ()
    calls: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BeaconIR:
    version: str
    entry: Dict[str, Any]  # entry file/qualname etc.
    nodes: Tuple["BeaconNode", ...]
    edges: Tuple["BeaconEdge", ...]
    symbols: Symbols
    forbidden: Tuple[NodeID, ...] = ()
    provenance: Dict[NodeID, Tuple[ProvenanceStep, ...]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)  # {"ir_hash": ..., "build_flags": ...}


# ----------------------------
# Constraints / verifier contracts
# ----------------------------

MatchSpec = Any  # contract: must be stable_json serializable (usually dict or dataclass)


@dataclass(frozen=True)
class Constraints:
    version: str
    required_symbols: Tuple[str, ...] = ()
    required_calls: Tuple[str, ...] = ()
    forbidden_specs: Tuple[MatchSpec, ...] = ()
    match_specs: Tuple[MatchSpec, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Violation:
    kind: str
    detail: str
    spec_ref: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Directive:
    action: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierReport:
    ok: bool
    coverage: Dict[str, int]
    violations: Tuple[Violation, ...] = ()
    directives: Tuple[Directive, ...] = ()
    meta: Dict[str, Any] = field(default_factory=dict)


# ----------------------------
# Runtime / build / run config contracts
# ----------------------------


@dataclass(frozen=True)
class ExecutionResult:
    status: str                 # "pass" | "fail" | "error"
    return_code: int
    stdout: str = ""
    stderr: str = ""
    trace: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BuildResult:
    ir: BeaconIR
    constraints: Constraints
    debug: Optional[Dict[str, Any]] = None


# NOTE: ModelConfig lives in llm/config.py; keep type as Any here to avoid hard coupling.
@dataclass(frozen=True)
class RunConfig:
    seed: int
    max_rounds: int
    use_verifier: bool
    outputs_dir: str
    reader: ReaderConfig
    model: Any  # expected: llm.config.ModelConfig
    adapter: Dict[str, Any] = field(default_factory=dict)