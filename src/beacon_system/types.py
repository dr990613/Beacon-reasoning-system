# src/beacon_system/types.py
# -*- coding: utf-8 -*-

"""
Unified data contracts for Beacon system.

Design goals:
- define all cross-module contracts in one place
- avoid implicit dict passing across modules
- keep contracts simple, stable, and explicit
- remain practical for current implementation stage

Notes:
- Most contracts provide `to_dict()` / `from_dict()` helpers
- Optional fields are used to keep early-stage integration tolerant
- These contracts are intended to gradually replace ad-hoc dicts
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict, is_dataclass
from typing import Any, Dict, List, Optional


# ============================================================
# Shared base helpers
# ============================================================

def _safe_asdict(obj: Any) -> Dict[str, Any]:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return dict(obj)
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}


def _list_or_empty(value: Optional[List[Any]]) -> List[Any]:
    return list(value) if value else []


def _dict_or_empty(value: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return dict(value) if value else {}


# ============================================================
# Input-side contracts
# ============================================================

# ============================================================
# Input-side contracts
# ============================================================

import json
import ast as py_ast


@dataclass
class TaskObject:
    """
    Unified task contract for heterogeneous benchmark/repo inputs.

    This contract is designed to absorb both Python and Java task records,
    especially CoderEval-like records whose raw fields are not fully aligned.

    Raw-source compatibility fields are preserved, then normalized into a
    stable internal view used by adapters / logic / pipeline.
    """

    # ---------- stable identity ----------
    task_id: Optional[str] = None
    lang: Optional[str] = None
    name: Optional[str] = None
    signature: Optional[str] = None
    docstring: Optional[str] = None
    instruction: Optional[str] = None
    human_label: Optional[str] = None

    # ---------- source location ----------
    file_path: Optional[str] = None
    file_name: Optional[str] = None
    target_file: Optional[str] = None
    project: Optional[str] = None
    package: Optional[str] = None
    class_name: Optional[str] = None
    qualname: Optional[str] = None
    entry_function: Optional[str] = None
    target_name: Optional[str] = None

    # ---------- raw source payload ----------
    code: str = ""
    file_content: str = ""
    class_level: str = ""
    all_context: str = ""
    dependency: str = ""

    # ---------- benchmark metadata ----------
    level: Optional[str] = None
    lineno: Optional[int] = None
    end_lineno: Optional[int] = None
    test_lineno: Optional[int] = None
    oracle_context: Dict[str, Any] = field(default_factory=dict)

    # ---------- normalized helper fields ----------
    imports: List[str] = field(default_factory=list)
    related_files: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---------- convenience normalized views ----------
    @property
    def source_text(self) -> str:
        """
        Main code body to be analyzed/generated.
        Priority:
            code > file_content > class_level
        """
        if self.code and self.code.strip():
            return self.code
        if self.file_content and self.file_content.strip():
            return self.file_content
        return self.class_level

    @property
    def context_text(self) -> str:
        """
        Extra context text around the task.
        Priority:
            all_context + class_level + file_content
        """
        parts = []
        if self.all_context.strip():
            parts.append(self.all_context)
        if self.class_level.strip():
            parts.append(self.class_level)
        if self.file_content.strip():
            parts.append(self.file_content)
        return "\n".join(parts).strip()

    @property
    def location_span(self) -> Dict[str, Optional[int]]:
        return {
            "lineno": self.lineno,
            "end_lineno": self.end_lineno,
            "test_lineno": self.test_lineno,
        }

    def normalize_lang(self) -> Optional[str]:
        """
        Light language normalization.
        """
        if not self.lang:
            if self.file_name and self.file_name.endswith(".java"):
                return "java"
            if self.file_path and self.file_path.endswith(".py"):
                return "python"
            if "public static" in self.code or "class " in self.code and ".java" in (self.file_name or ""):
                return "java"
            if "def " in self.code or "import " in self.all_context:
                return "python"
            return None

        value = str(self.lang).strip().lower()
        if value in {"py", "python"}:
            return "python"
        if value in {"java"}:
            return "java"
        return value

    def normalized_target_name(self) -> Optional[str]:
        return self.target_name or self.name

    def normalized_target_file(self) -> Optional[str]:
        return self.target_file or self.file_path or self.file_name

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["lang"] = self.normalize_lang()
        data["target_name"] = self.normalized_target_name()
        data["target_file"] = self.normalized_target_file()
        data["source_text"] = self.source_text
        data["context_text"] = self.context_text
        data["location_span"] = self.location_span
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TaskObject":
        """
        Generic tolerant constructor.
        """
        return cls(
            task_id=data.get("task_id") or data.get("_id"),
            lang=data.get("lang"),
            name=data.get("name"),
            signature=data.get("signature"),
            docstring=data.get("docstring"),
            instruction=data.get("instruction"),
            human_label=data.get("human_label"),

            file_path=data.get("file_path"),
            file_name=data.get("file_name"),
            target_file=data.get("target_file"),
            project=data.get("project"),
            package=data.get("package"),
            class_name=data.get("class_name"),
            qualname=data.get("qualname"),
            entry_function=data.get("entry_function"),
            target_name=data.get("target_name"),

            code=data.get("code", "") or "",
            file_content=data.get("file_content", "") or "",
            class_level=data.get("class_level", "") or "",
            all_context=data.get("all_context", "") or "",
            dependency=data.get("dependency", "") or "",

            level=data.get("level"),
            lineno=_to_int_or_none(data.get("lineno")),
            end_lineno=_to_int_or_none(data.get("end_lineno")),
            test_lineno=_to_int_or_none(data.get("test_lineno")),
            oracle_context=_parse_loose_context(data.get("oracle_context")),

            imports=_list_or_empty(data.get("imports")),
            related_files=_list_or_empty(data.get("related_files")),
            metadata=_dict_or_empty(data.get("metadata")),
        )

    @classmethod
    def from_codereval_record(cls, record: Dict[str, Any]) -> "TaskObject":
        """
        Specialized normalizer for CoderEval-style records.

        Handles both Python and Java record formats.
        """
        task = cls.from_dict(record)

        # ---- infer language from record shape ----
        inferred_lang = task.normalize_lang()
        task.lang = inferred_lang

        # ---- normalize Java class-level context ----
        if inferred_lang == "java":
            if not task.file_path and task.file_name:
                task.file_path = task.file_name
            if not task.target_name and task.name:
                task.target_name = task.name
            if not task.target_file:
                task.target_file = task.file_name or task.file_path

            # For Java tasks, class_level may be absent as a dedicated field.
            # We keep all_context as the primary class-level / import context.
            if not task.class_level and task.all_context:
                task.class_level = task.all_context

            if task.class_name and task.name:
                task.qualname = f"{task.class_name}.{task.name}"

        # ---- normalize Python context ----
        elif inferred_lang == "python":
            if not task.target_name and task.name:
                task.target_name = task.name
            if not task.target_file:
                task.target_file = task.file_path or task.file_name

            # Python all_context is often a serialized object like:
            # { "import": "...", "file": "", "class": "" }
            ctx = _parse_loose_context(task.all_context)
            if ctx:
                task.metadata.setdefault("parsed_all_context", ctx)

                import_part = str(ctx.get("import", "") or "")
                file_part = str(ctx.get("file", "") or "")
                class_part = str(ctx.get("class", "") or "")

                if import_part:
                    task.imports.extend(_split_context_tokens(import_part))

                merged_class_level = "\n".join(
                    part for part in [class_part, file_part] if part.strip()
                ).strip()
                if merged_class_level and not task.class_level.strip():
                    task.class_level = merged_class_level

        # ---- normalize metadata ----
        task.metadata.setdefault("raw_record_type", "codereval")
        task.metadata.setdefault("level", task.level)
        task.metadata.setdefault("project", task.project)
        task.metadata.setdefault("package", task.package)
        task.metadata.setdefault("class_name", task.class_name)
        task.metadata.setdefault("human_label", task.human_label)
        task.metadata.setdefault("dependency", task.dependency)

        return task


@dataclass
class ProjectFile:
    path: str
    content: str
    lang: Optional[str] = None
    package: Optional[str] = None
    class_name: Optional[str] = None
    symbols: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProjectIndex:
    """
    Project-side context contract.

    This contract is intentionally lightweight:
    it stores repository/program-level context, but does not force
    Python and Java into the same source-layout assumptions.
    """
    project_root: Optional[str] = None
    repo_name: Optional[str] = None
    lang: Optional[str] = None

    files: List[ProjectFile] = field(default_factory=list)
    symbol_index: Dict[str, Any] = field(default_factory=dict)
    entry_candidates: List[str] = field(default_factory=list)

    # lightweight language-specific indexes
    python_modules: Dict[str, Any] = field(default_factory=dict)
    java_classes: Dict[str, Any] = field(default_factory=dict)

    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_root": self.project_root,
            "repo_name": self.repo_name,
            "lang": self.lang,
            "files": [
                f.to_dict() if isinstance(f, ProjectFile) else _safe_asdict(f)
                for f in self.files
            ],
            "symbol_index": dict(self.symbol_index),
            "entry_candidates": list(self.entry_candidates),
            "python_modules": dict(self.python_modules),
            "java_classes": dict(self.java_classes),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProjectIndex":
        files = []
        for item in data.get("files", []) or []:
            if isinstance(item, ProjectFile):
                files.append(item)
            else:
                files.append(
                    ProjectFile(
                        path=item.get("path", ""),
                        content=item.get("content", ""),
                        lang=item.get("lang"),
                        package=item.get("package"),
                        class_name=item.get("class_name"),
                        symbols=_list_or_empty(item.get("symbols")),
                        metadata=_dict_or_empty(item.get("metadata")),
                    )
                )

        return cls(
            project_root=data.get("project_root"),
            repo_name=data.get("repo_name"),
            lang=data.get("lang"),
            files=files,
            symbol_index=_dict_or_empty(data.get("symbol_index")),
            entry_candidates=_list_or_empty(data.get("entry_candidates")),
            python_modules=_dict_or_empty(data.get("python_modules")),
            java_classes=_dict_or_empty(data.get("java_classes")),
            metadata=_dict_or_empty(data.get("metadata")),
        )


# ============================================================
# Input-side helpers
# ============================================================

def _to_int_or_none(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except Exception:
        return None


def _parse_loose_context(value: Any) -> Dict[str, Any]:
    """
    Parse loose JSON / Python-literal-like context strings safely.

    Examples:
    - '{ "apis" : "[isEmpty, trim]", "classes" : "[String[], String]" }'
    - "{ 'apis' : \"['localize', 'map']\", 'vars' : '[]' }"
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)

    text = str(value).strip()
    if not text:
        return {}

    # first try json
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # then try python literal
    try:
        obj = py_ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    return {}


def _split_context_tokens(text: str) -> List[str]:
    """
    Split loose import/token context into a simple list.
    Example:
        "time datetime pytz datetime" -> ["time", "datetime", "pytz"]
    """
    tokens = []
    for item in re.split(r"[\s,]+", text.strip()):
        item = item.strip()
        if item and item not in tokens:
            tokens.append(item)
    return tokens


# ============================================================
# Raw Beacon IR contracts
# ============================================================

@dataclass
class RawIRNode:
    function_name: str
    line_no: int
    code: str
    kind: str = "unknown"
    roles: List[str] = field(default_factory=list)
    source: str = "local"   # local / global / synthetic
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RawIREdge:
    src_function: str
    src_line_no: int
    dst_function: str
    dst_line_no: int
    edge_type: str          # depends_on / call / return_flow / global_state
    label: str = ""
    rule: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RawBeaconFunction:
    function_name: str
    signature: Optional[str] = None
    lang: Optional[str] = None
    local_beacon_node_ids: List[str] = field(default_factory=list)
    global_beacon_node_ids: List[str] = field(default_factory=list)
    output_node_ids: List[str] = field(default_factory=list)
    nodes: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RawBeaconIR:
    lang: str
    entry_functions: List[str] = field(default_factory=list)
    functions: List[RawBeaconFunction] = field(default_factory=list)
    nodes: List[RawIRNode] = field(default_factory=list)
    edges: List[RawIREdge] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lang": self.lang,
            "entry_functions": list(self.entry_functions),
            "functions": [f.to_dict() if isinstance(f, RawBeaconFunction) else _safe_asdict(f) for f in self.functions],
            "nodes": [n.to_dict() if isinstance(n, RawIRNode) else _safe_asdict(n) for n in self.nodes],
            "edges": [e.to_dict() if isinstance(e, RawIREdge) else _safe_asdict(e) for e in self.edges],
            "provenance": dict(self.provenance),
            "debug": dict(self.debug),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RawBeaconIR":
        return cls(
            lang=data.get("lang", "unknown"),
            entry_functions=_list_or_empty(data.get("entry_functions")),
            functions=[
                item if isinstance(item, RawBeaconFunction) else RawBeaconFunction(
                    function_name=item.get("function_name", ""),
                    signature=item.get("signature"),
                    lang=item.get("lang"),
                    local_beacon_node_ids=_list_or_empty(item.get("local_beacon_node_ids")),
                    global_beacon_node_ids=_list_or_empty(item.get("global_beacon_node_ids")),
                    output_node_ids=_list_or_empty(item.get("output_node_ids")),
                    nodes=_list_or_empty(item.get("nodes")),
                )
                for item in data.get("functions", []) or []
            ],
            nodes=[
                item if isinstance(item, RawIRNode) else RawIRNode(
                    function_name=item.get("function_name", ""),
                    line_no=int(item.get("line_no", -1)),
                    code=item.get("code", ""),
                    kind=item.get("kind", "unknown"),
                    roles=_list_or_empty(item.get("roles")),
                    source=item.get("source", "local"),
                    metadata=_dict_or_empty(item.get("metadata")),
                )
                for item in data.get("nodes", []) or []
            ],
            edges=[
                item if isinstance(item, RawIREdge) else RawIREdge(
                    src_function=item.get("src_function", ""),
                    src_line_no=int(item.get("src_line_no", -1)),
                    dst_function=item.get("dst_function", ""),
                    dst_line_no=int(item.get("dst_line_no", -1)),
                    edge_type=item.get("edge_type", ""),
                    label=item.get("label", ""),
                    rule=item.get("rule", ""),
                    metadata=_dict_or_empty(item.get("metadata")),
                )
                for item in data.get("edges", []) or []
            ],
            provenance=_dict_or_empty(data.get("provenance")),
            debug=_dict_or_empty(data.get("debug")),
        )


# ============================================================
# Beacon Tree contracts
# ============================================================

@dataclass
class TreeStatement:
    function_name: str
    line_no: int
    code: str
    children: List[Dict[str, Any]] = field(default_factory=list)
    visited_ref: bool = False
    visited_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FunctionTree:
    function_name: str
    signature: Optional[str] = None
    root_statements: List[TreeStatement] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "signature": self.signature,
            "root_statements": [
                s.to_dict() if isinstance(s, TreeStatement) else _safe_asdict(s)
                for s in self.root_statements
            ],
        }


@dataclass
class BeaconTree:
    entry_function: Optional[str] = None
    functions: List[FunctionTree] = field(default_factory=list)
    rendered_text: str = ""
    refiner_notes: List[str] = field(default_factory=list)
    refiner_audit: Dict[str, Any] = field(default_factory=dict)
    refiner_warnings: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_function": self.entry_function,
            "functions": [f.to_dict() if isinstance(f, FunctionTree) else _safe_asdict(f) for f in self.functions],
            "rendered_text": self.rendered_text,
            "refiner_notes": list(self.refiner_notes),
            "refiner_audit": dict(self.refiner_audit),
            "refiner_warnings": list(self.refiner_warnings),
            "debug": dict(self.debug),
        }


# ============================================================
# Signature Hints contracts
# ============================================================

@dataclass
class ParameterHint:
    name: str
    kind: Optional[str] = None
    type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CallHint:
    line_no: int
    callee: str
    arguments: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VariableOriginHint:
    variable: str
    line_no: int
    origin: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ReturnHint:
    line_no: int
    value: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FunctionSignatureHints:
    function_name: str
    signature: Optional[str] = None
    parameters: List[ParameterHint] = field(default_factory=list)
    call_hints: List[CallHint] = field(default_factory=list)
    variable_origins: List[VariableOriginHint] = field(default_factory=list)
    return_hints: List[ReturnHint] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "function_name": self.function_name,
            "signature": self.signature,
            "parameters": [p.to_dict() if isinstance(p, ParameterHint) else _safe_asdict(p) for p in self.parameters],
            "call_hints": [c.to_dict() if isinstance(c, CallHint) else _safe_asdict(c) for c in self.call_hints],
            "variable_origins": [v.to_dict() if isinstance(v, VariableOriginHint) else _safe_asdict(v) for v in self.variable_origins],
            "return_hints": [r.to_dict() if isinstance(r, ReturnHint) else _safe_asdict(r) for r in self.return_hints],
        }


@dataclass
class SignatureHints:
    lang: str
    functions: List[FunctionSignatureHints] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lang": self.lang,
            "functions": [f.to_dict() if isinstance(f, FunctionSignatureHints) else _safe_asdict(f) for f in self.functions],
            "debug": dict(self.debug),
        }


# ============================================================
# Constraint Summary contract
# ============================================================

@dataclass
class ConstraintSummary:
    required_functions: List[str] = field(default_factory=list)
    key_calls: List[str] = field(default_factory=list)
    edge_types_present: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Logic result contract
# ============================================================

@dataclass
class LogicBuildResult:
    raw_ir: RawBeaconIR
    beacon_tree: BeaconTree
    signature_hints: SignatureHints
    constraint_summary: ConstraintSummary
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_ir": self.raw_ir.to_dict() if isinstance(self.raw_ir, RawBeaconIR) else _safe_asdict(self.raw_ir),
            "beacon_tree": self.beacon_tree.to_dict() if isinstance(self.beacon_tree, BeaconTree) else _safe_asdict(self.beacon_tree),
            "signature_hints": self.signature_hints.to_dict() if isinstance(self.signature_hints, SignatureHints) else _safe_asdict(self.signature_hints),
            "constraint_summary": self.constraint_summary.to_dict() if isinstance(self.constraint_summary, ConstraintSummary) else _safe_asdict(self.constraint_summary),
            "debug": dict(self.debug),
        }


# ============================================================
# Agent-side result contracts
# ============================================================

@dataclass
class GenerationResult:
    accepted: bool = False
    language: Optional[str] = None
    target_function: Optional[str] = None
    generated_code: str = ""
    prompt_snapshot: Optional[str] = None
    raw_response: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RebuildResult:
    accepted: bool = False
    patched_program: str = ""
    patched_file_path: Optional[str] = None
    rebuilt_logic: Optional[LogicBuildResult] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "patched_program": self.patched_program,
            "patched_file_path": self.patched_file_path,
            "rebuilt_logic": (
                self.rebuilt_logic.to_dict()
                if isinstance(self.rebuilt_logic, LogicBuildResult)
                else _safe_asdict(self.rebuilt_logic) if self.rebuilt_logic is not None else None
            ),
            "diagnostics": dict(self.diagnostics),
            "warnings": list(self.warnings),
            "debug": dict(self.debug),
        }


@dataclass
class VerificationIssue:
    category: str
    message: str
    function_name: Optional[str] = None
    line_no: Optional[int] = None
    severity: str = "warning"
    advice: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VerificationResult:
    accepted: bool = False
    issues: List[VerificationIssue] = field(default_factory=list)
    revision_advice: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accepted": self.accepted,
            "issues": [i.to_dict() if isinstance(i, VerificationIssue) else _safe_asdict(i) for i in self.issues],
            "revision_advice": list(self.revision_advice),
            "warnings": list(self.warnings),
            "debug": dict(self.debug),
        }


# ============================================================
# Full run trace contract
# ============================================================

@dataclass
class RunTrace:
    task: TaskObject
    logic: LogicBuildResult

    generation_round_1: Optional[GenerationResult] = None
    rebuild_round_1: Optional[RebuildResult] = None
    verification_round_1: Optional[VerificationResult] = None

    generation_round_2: Optional[GenerationResult] = None
    rebuild_round_2: Optional[RebuildResult] = None
    verification_round_2: Optional[VerificationResult] = None

    final_code: Optional[str] = None
    final_status: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    debug: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task": self.task.to_dict() if isinstance(self.task, TaskObject) else _safe_asdict(self.task),
            "logic": self.logic.to_dict() if isinstance(self.logic, LogicBuildResult) else _safe_asdict(self.logic),
            "generation_round_1": (
                self.generation_round_1.to_dict()
                if isinstance(self.generation_round_1, GenerationResult)
                else _safe_asdict(self.generation_round_1) if self.generation_round_1 is not None else None
            ),
            "rebuild_round_1": (
                self.rebuild_round_1.to_dict()
                if isinstance(self.rebuild_round_1, RebuildResult)
                else _safe_asdict(self.rebuild_round_1) if self.rebuild_round_1 is not None else None
            ),
            "verification_round_1": (
                self.verification_round_1.to_dict()
                if isinstance(self.verification_round_1, VerificationResult)
                else _safe_asdict(self.verification_round_1) if self.verification_round_1 is not None else None
            ),
            "generation_round_2": (
                self.generation_round_2.to_dict()
                if isinstance(self.generation_round_2, GenerationResult)
                else _safe_asdict(self.generation_round_2) if self.generation_round_2 is not None else None
            ),
            "rebuild_round_2": (
                self.rebuild_round_2.to_dict()
                if isinstance(self.rebuild_round_2, RebuildResult)
                else _safe_asdict(self.rebuild_round_2) if self.rebuild_round_2 is not None else None
            ),
            "verification_round_2": (
                self.verification_round_2.to_dict()
                if isinstance(self.verification_round_2, VerificationResult)
                else _safe_asdict(self.verification_round_2) if self.verification_round_2 is not None else None
            ),
            "final_code": self.final_code,
            "final_status": self.final_status,
            "metadata": dict(self.metadata),
            "debug": dict(self.debug),
        }