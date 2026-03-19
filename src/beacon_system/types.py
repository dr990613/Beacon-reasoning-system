# src/beacon_system/types.py
# -*- coding: utf-8 -*-

"""
Contracts (single source of truth)

All cross-module data contracts live here to avoid drift.

Design rules:
- Use frozen dataclasses where possible to reduce accidental mutation.
- Prefer tuple[...] for stable ordering / deterministic serialization.
- All objects should be stable-json serializable.
- Keep this file dependency-light: do not import logic / agent / adapter implementations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

# ============================================================
# Common aliases
# ============================================================

JSONDict = Dict[str, Any]
StrMap = Dict[str, str]
NodeID = str
MatchSpec = Any  # Must be stable-json serializable (usually dict / tuple / scalar)


# ============================================================
# Core task / project contracts
# ============================================================


@dataclass(frozen=True)
class TaskTarget:
    """
    Concrete code target to modify.

    file:
        Relative path inside repo/project root.

    qualname:
        Qualified symbol name, e.g. "ClassName.method" or "func_name".
        For file-level tasks, qualname can be an empty string.
    """
    file: str
    qualname: str = ""


@dataclass(frozen=True)
class TaskObject:
    """
    Standard task input consumed by pipeline / agents / logic.

    Notes:
    - `target` is kept as a dict to match the architecture contract text exactly.
    - `level` is intentionally plain str to stay benchmark-tolerant.
      Suggested values: "function" | "file" | "project".
    """
    id: str
    lang: str
    level: str
    target: StrMap  # {"file": "...", "qualname": "..."}
    spec: str = ""
    context: JSONDict = field(default_factory=dict)
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class ProjectIndex:
    """
    Adapter-built project snapshot, treated as read-only by logic / agents.

    IMPORTANT:
    - `entry_file` / `entry_qualname` are the canonical reasoning entrypoint.
    - `files` is a mapping {relative_file_path: source_code_str}.
    - `ast_index` / `symbols` / `callgraph` are intentionally Any-friendly
      so adapters can evolve without breaking contracts.
    """
    root: str
    entry_file: str
    entry_qualname: str
    files: Dict[str, str]
    ast_index: JSONDict = field(default_factory=dict)
    symbols: JSONDict = field(default_factory=dict)
    callgraph: JSONDict = field(default_factory=dict)
    meta: JSONDict = field(default_factory=dict)


# ============================================================
# Logic engine config + result contracts
# ============================================================


@dataclass(frozen=True)
class ReaderConfig:
    """
    Configuration for deterministic Beacon reasoning.

    enable_global:
        Whether to run global reasoning rules.

    validation_filter:
        Whether to run validation / pruning step.

    max_local_nodes:
        Optional cap for local graph size.

    max_global_inline:
        Optional cap for global inlining / call expansion.
    """
    enable_global: bool = True
    validation_filter: bool = True
    max_local_nodes: Optional[int] = None
    max_global_inline: Optional[int] = None


@dataclass(frozen=True)
class Anchor:
    """
    Source anchor for a node or evidence location.
    """
    file: str
    qualname: str
    lineno: int
    col: int
    end_lineno: int
    end_col: int
    namespace: str  # e.g. LOCAL / GLOBAL / MODULE / CLASS / FUNCTION


@dataclass(frozen=True)
class ProvenanceStep:
    """
    One reasoning step that explains how a Beacon node/constraint was derived.
    """
    rule_id: str          # e.g. "L-OUT" / "L-DEP" / "G-CALL"
    source: str           # source anchor id / node id / symbolic reference
    note: str = ""


@dataclass(frozen=True)
class BeaconNode:
    """
    Minimal normalized reasoning node exposed to agent side.
    """
    id: NodeID
    kind: str
    text: str
    anchor: Anchor
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class BeaconEdge:
    """
    Minimal normalized reasoning edge exposed to agent side.
    """
    kind: str             # e.g. "data" | "control" | "call"
    src: NodeID
    dst: NodeID
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class Symbols:
    """
    Coarse symbol summary extracted by logic / adapter.
    """
    imports: Tuple[str, ...] = ()
    globals: Tuple[str, ...] = ()
    attrs: Tuple[str, ...] = ()
    calls: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BeaconIR:
    """
    Deterministic Beacon reasoning output consumed by planning / generator / checks.

    entry:
        Canonical entry descriptor, typically containing file / qualname / lang / task_id.

    forbidden:
        Node ids or symbolic ids that should not be reconstructed / used directly.
    """
    version: str
    entry: JSONDict
    nodes: Tuple[BeaconNode, ...]
    edges: Tuple[BeaconEdge, ...]
    symbols: Symbols
    forbidden: Tuple[NodeID, ...] = ()
    provenance: Dict[NodeID, Tuple[ProvenanceStep, ...]] = field(default_factory=dict)
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class Constraints:
    """
    Agent-facing constraints derived from logic.

    required_symbols:
        Symbols that should appear or be respected structurally.

    required_calls:
        Calls that should appear or be preserved.

    forbidden_specs:
        Patterns / rules that must not be produced.

    match_specs:
        Positive structural expectations for verifier.
    """
    version: str
    required_symbols: Tuple[str, ...] = ()
    required_calls: Tuple[str, ...] = ()
    forbidden_specs: Tuple[MatchSpec, ...] = ()
    match_specs: Tuple[MatchSpec, ...] = ()
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class BuildResult:
    """
    Output of logic.engine.build(...).
    """
    ir: BeaconIR
    constraints: Constraints
    debug: Optional[JSONDict] = None


# ============================================================
# Strict code-generation format contracts
# ============================================================


@dataclass(frozen=True)
class CodeBlock:
    """
    One concrete code artifact returned by generator.

    language:
        Expected language tag, e.g. "python", "java".

    content:
        Raw code only. No markdown fence, no explanation text.

    kind:
        What this block represents, e.g.:
        - "replacement_impl"
        - "full_file"
        - "helper"
        - "test"
    """
    language: str
    content: str
    kind: str = "replacement_impl"
    filename: str = ""
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationPayload:
    """
    Strict generator output contract.
    """
    primary: CodeBlock
    auxiliary: Tuple[CodeBlock, ...] = ()
    format_ok: bool = True
    raw_text: str = ""
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class OutputFormatSpec:
    """
    Hard output format requirement passed into generator/checker.
    """
    code_only: bool = True
    fenced_code_block: bool = False
    single_block_only: bool = True
    require_language_match: bool = True
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class FormatValidationResult:
    """
    Result of validating raw model output against output-format contract.
    """
    ok: bool
    normalized_code: str = ""
    issues: Tuple[str, ...] = ()
    meta: JSONDict = field(default_factory=dict)


# ============================================================
# Agent planning / checking / verification contracts
# ============================================================


@dataclass(frozen=True)
class ThoughtCandidate:
    """
    One implementation thought generated by planning.
    """
    id: str
    text: str
    rationale: str = ""
    steps: Tuple[str, ...] = ()
    assumptions: Tuple[str, ...] = ()
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class ThoughtScore:
    """
    Scoring result for a planning candidate.
    """
    thought_id: str
    total: float
    subscores: JSONDict = field(default_factory=dict)
    reasons: Tuple[str, ...] = ()
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class LogicAcceptanceReport:
    """
    Result of agent-side acceptance checks on logic outputs.
    """
    ok: bool
    issues: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class BeaconUsageReport:
    """
    Evidence that generator output actually used Beacon artifacts.
    """
    ok: bool
    used_required_symbols: Tuple[str, ...] = ()
    used_required_calls: Tuple[str, ...] = ()
    missing_symbols: Tuple[str, ...] = ()
    missing_calls: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class Violation:
    """
    One verifier failure item.
    """
    kind: str
    detail: str
    spec_ref: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class Directive:
    """
    One revise instruction emitted by verifier/checks.
    """
    action: str
    payload: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class VerifierResult:
    """
    Output of agent verifier.
    """
    ok: bool
    coverage: JSONDict = field(default_factory=dict)
    violations: Tuple[Violation, ...] = ()
    directives: Tuple[Directive, ...] = ()
    meta: JSONDict = field(default_factory=dict)


# Backward-compatible alias for older code paths.
VerifierReport = VerifierResult


# ============================================================
# Runtime / patch / execution contracts
# ============================================================


@dataclass(frozen=True)
class PatchTarget:
    """
    Concrete patch placement used by runtime/patcher.
    """
    file: str
    qualname: str = ""


@dataclass(frozen=True)
class PatchResult:
    """
    Result of writing generated code back into workspace copy.
    """
    ok: bool
    target: PatchTarget
    applied: bool = False
    backup_path: str = ""
    output_path: str = ""
    detail: str = ""
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class ExecResult:
    """
    Runtime execution result.

    status:
        Suggested values: "pass" | "fail" | "error" | "timeout"
    """
    status: str
    return_code: int
    stdout: str = ""
    stderr: str = ""
    trace: str = ""
    metrics: JSONDict = field(default_factory=dict)


# Backward-compatible alias for older code paths.
ExecutionResult = ExecResult


# ============================================================
# Memory contracts
# ============================================================


@dataclass(frozen=True)
class MemoryRecord:
    """
    One experience-memory item.
    """
    key: str
    value: JSONDict
    source_run_id: str = ""
    tags: Tuple[str, ...] = ()
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryReadResult:
    """
    Output of memory read.
    """
    items: Tuple[MemoryRecord, ...] = ()
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryWriteResult:
    """
    Output of memory write.
    """
    written: Tuple[MemoryRecord, ...] = ()
    skipped: Tuple[str, ...] = ()
    meta: JSONDict = field(default_factory=dict)


# ============================================================
# Config contracts
# ============================================================


@dataclass(frozen=True)
class AgentConfig:
    """
    Agent-side execution policy.
    """
    max_rounds: int = 2
    max_thoughts: int = 3
    keep_top_k: int = 1
    use_memory: bool = True
    use_verifier: bool = True
    require_logic_acceptance: bool = True
    require_beacon_usage_check: bool = True
    output_format: OutputFormatSpec = field(default_factory=OutputFormatSpec)
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class RuntimeConfig:
    """
    Runtime adapter config.
    """
    work_dir: str = ""
    run_command: Tuple[str, ...] = ()
    env: JSONDict = field(default_factory=dict)
    timeout_sec: Optional[int] = None
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class RunConfig:
    """
    Top-level runtime config consumed by pipeline.
    """
    seed: int
    outputs_dir: str
    reader: ReaderConfig
    model: Any
    agent: AgentConfig = field(default_factory=AgentConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    adapter: JSONDict = field(default_factory=dict)
    meta: JSONDict = field(default_factory=dict)


# ============================================================
# Full round / pipeline result contracts
# ============================================================


@dataclass(frozen=True)
class GenerationRoundResult:
    """
    One agent generation round snapshot.
    """
    round_index: int
    selected_thought_id: str = ""
    generation: Optional[GenerationPayload] = None
    format_check: Optional[FormatValidationResult] = None
    verifier: Optional[VerifierResult] = None
    beacon_usage: Optional[BeaconUsageReport] = None
    exec_result: Optional[ExecResult] = None
    directives: Tuple[Directive, ...] = ()
    meta: JSONDict = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineResult:
    """
    Final pipeline output for one task run.
    """
    task: TaskObject
    build: BuildResult
    rounds: Tuple[GenerationRoundResult, ...] = ()
    final_generation: Optional[GenerationPayload] = None
    final_verifier: Optional[VerifierResult] = None
    final_exec: Optional[ExecResult] = None
    success: bool = False
    meta: JSONDict = field(default_factory=dict)