# src/beacon_system/logic/rules_local.py
# -*- coding: utf-8 -*-

"""
Local Beacon Logic rules.

Responsibilities:
- work ONLY within function / method boundary
- support Python and Java
- implement local beacon extraction as:
    1) observable output detection
    2) backward dependency closure
    3) heuristic validation filtering
    4) reduction / normalization

Non-goals:
- no interprocedural propagation
- no global beacon merging
- no planner-like reasoning

Input:
- preprocessed dict from logic.preprocess.preprocess_task(...)
- or any dict-like/object-like source that contains:
    lang, body_text, file_content_text, class_level_text, target_name, target_signature

Output:
- a stable dict describing local beacons per function/method
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import ast
import re
from typing import Any, Dict, List, Optional, Set, Tuple


# ============================================================
# Small local contracts
# ============================================================

@dataclass
class LocalBeaconNode:
    node_id: str
    function_name: str
    line_no: int
    kind: str
    code: str
    roles: List[str]
    depends_on: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocalFunctionBeacon:
    function_name: str
    signature: Optional[str]
    lang: str
    output_node_ids: List[str]
    beacon_node_ids: List[str]
    filtered_validation_node_ids: List[str]
    nodes: List[Dict[str, Any]]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class LocalRulesResult:
    lang: str
    functions: List[Dict[str, Any]]
    warnings: List[str]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Public API
# ============================================================

def build_local_beacons(preprocessed: Any) -> Dict[str, Any]:
    """
    Build local beacons for Python / Java source.

    Expected input: result from preprocess.preprocess_task(...)
    """
    data = _as_dict(preprocessed)
    lang = str(data.get("lang", "unknown")).lower().strip()
    body_text = _read_text(data, "body_text")
    file_text = _read_text(data, "file_content_text")
    class_level = _read_text(data, "class_level_text")
    target_name = _read_optional_text(data, "target_name")
    target_signature = _read_optional_text(data, "target_signature")

    source = body_text or file_text or class_level
    warnings: List[str] = []

    if lang == "python":
        functions, debug, extra_warnings = _build_local_python(
            source=source,
            target_name=target_name,
            target_signature=target_signature,
        )
        warnings.extend(extra_warnings)
    elif lang == "java":
        functions, debug, extra_warnings = _build_local_java(
            source=source,
            class_level=class_level,
            target_name=target_name,
            target_signature=target_signature,
        )
        warnings.extend(extra_warnings)
    else:
        functions = []
        debug = {"reason": "unsupported language"}
        warnings.append(f"unsupported language '{lang}' in local rules")

    return LocalRulesResult(
        lang=lang,
        functions=[f.to_dict() for f in functions],
        warnings=warnings,
        debug=debug,
    ).to_dict()


# ============================================================
# Python local logic
# ============================================================

def _build_local_python(
    source: str,
    target_name: Optional[str],
    target_signature: Optional[str],
) -> Tuple[List[LocalFunctionBeacon], Dict[str, Any], List[str]]:
    warnings: List[str] = []

    if not source.strip():
        return [], {"python_parse": "empty_source"}, ["empty python source"]

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [], {"python_parse": "syntax_error", "detail": str(exc)}, [
            "python source parse failed in local rules"
        ]

    lines = source.splitlines()
    results: List[LocalFunctionBeacon] = []

    func_nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]

    if not func_nodes:
        warnings.append("no python function found; local rules returned empty result")

    for fn in func_nodes:
        if target_name and fn.name != target_name:
            # keep all functions if target_name absent; otherwise prefer exact target only
            continue

        index = _PythonFunctionIndex(fn=fn, source_lines=lines)
        index.build()

        signature = _python_signature_text(fn, lines)
        if target_name and fn.name == target_name and target_signature:
            signature = target_signature

        output_ids = index.find_output_nodes()
        beacon_ids = index.backward_closure(output_ids)
        filtered_ids = index.filter_validation_nodes(beacon_ids, output_ids)
        reduced_ids = index.reduce_nodes(filtered_ids)

        nodes = [
            index.to_beacon_node(nid).to_dict()
            for nid in reduced_ids
        ]

        results.append(
            LocalFunctionBeacon(
                function_name=fn.name,
                signature=signature,
                lang="python",
                output_node_ids=sorted(output_ids),
                beacon_node_ids=sorted(reduced_ids),
                filtered_validation_node_ids=sorted(set(beacon_ids) - set(filtered_ids)),
                nodes=nodes,
                warnings=index.warnings.copy(),
            )
        )

    debug = {
        "function_count": len(func_nodes),
        "selected_count": len(results),
    }
    return results, debug, warnings


class _PythonFunctionIndex:
    """
    Lightweight local dependence extractor for a Python function.

    Strategy:
    - one node per statement-level AST element
    - build assignment/use index
    - approximate backward data/control dependencies
    """

    def __init__(self, fn: ast.AST, source_lines: List[str]) -> None:
        self.fn = fn
        self.source_lines = source_lines

        self.nodes_by_id: Dict[str, ast.AST] = {}
        self.code_by_id: Dict[str, str] = {}
        self.kind_by_id: Dict[str, str] = {}
        self.roles_by_id: Dict[str, Set[str]] = {}
        self.line_by_id: Dict[str, int] = {}
        self.depends_on: Dict[str, Set[str]] = {}
        self.var_def_line_to_node: Dict[int, str] = {}
        self.latest_def_by_name: Dict[str, List[Tuple[int, str]]] = {}
        self.control_stack_by_node: Dict[str, List[str]] = {}
        self.warnings: List[str] = []

    def build(self) -> None:
        for node in ast.walk(self.fn):
            if self._is_statement_like(node):
                nid = self._node_id(node)
                self.nodes_by_id[nid] = node
                self.code_by_id[nid] = self._code_of(node)
                self.kind_by_id[nid] = type(node).__name__.lower()
                self.roles_by_id[nid] = set()
                self.line_by_id[nid] = getattr(node, "lineno", -1)
                self.depends_on[nid] = set()

        self._walk_body_with_control(self.fn.body, active_controls=[])
        self._build_data_dependencies()

    def find_output_nodes(self) -> Set[str]:
        outputs: Set[str] = set()
        for nid, node in self.nodes_by_id.items():
            if self._is_output_node(node):
                outputs.add(nid)
                self.roles_by_id[nid].add("output")
        return outputs

    def backward_closure(self, seed_ids: Set[str]) -> Set[str]:
        visited = set(seed_ids)
        worklist = list(seed_ids)

        while worklist:
            nid = worklist.pop()
            for dep in sorted(self.depends_on.get(nid, set())):
                if dep not in visited:
                    visited.add(dep)
                    worklist.append(dep)
        return visited

    def filter_validation_nodes(self, beacon_ids: Set[str], output_ids: Set[str]) -> Set[str]:
        kept = set(beacon_ids)
        for nid in list(beacon_ids):
            node = self.nodes_by_id[nid]
            if self._looks_like_validation(node):
                # keep validation node if it is itself an observable output or direct return guard
                if nid in output_ids:
                    continue
                kept.discard(nid)
                self.roles_by_id[nid].add("validation_filtered")
        return kept

    def reduce_nodes(self, beacon_ids: Set[str]) -> List[str]:
        """
        Keep statement nodes only, remove duplicates by normalized code, sort by line.
        """
        ordered = sorted(beacon_ids, key=lambda x: (self.line_by_id.get(x, 10**9), x))
        reduced: List[str] = []
        seen_norm: Set[Tuple[int, str]] = set()

        for nid in ordered:
            code = self.code_by_id.get(nid, "").strip()
            norm = _normalize_code(code)
            key = (self.line_by_id.get(nid, -1), norm)
            if key in seen_norm:
                continue
            seen_norm.add(key)
            reduced.append(nid)
        return reduced

    def to_beacon_node(self, nid: str) -> LocalBeaconNode:
        return LocalBeaconNode(
            node_id=nid,
            function_name=getattr(self.fn, "name", "<lambda>"),
            line_no=self.line_by_id.get(nid, -1),
            kind=self.kind_by_id.get(nid, "unknown"),
            code=self.code_by_id.get(nid, ""),
            roles=sorted(self.roles_by_id.get(nid, set())),
            depends_on=sorted(self.depends_on.get(nid, set())),
        )

    # -------------------------
    # build internals
    # -------------------------

    def _walk_body_with_control(self, body: List[ast.stmt], active_controls: List[str]) -> None:
        for stmt in body:
            if not self._is_statement_like(stmt):
                continue

            nid = self._node_id(stmt)
            self.control_stack_by_node[nid] = list(active_controls)

            for ctrl_id in active_controls:
                self.depends_on[nid].add(ctrl_id)

            if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
                self.roles_by_id[nid].add("control")

            if self._is_definition_stmt(stmt):
                defs = self._defined_names(stmt)
                for name in defs:
                    self.latest_def_by_name.setdefault(name, []).append((getattr(stmt, "lineno", -1), nid))
                    self.roles_by_id[nid].add("definition")

            if isinstance(stmt, ast.Return):
                self.roles_by_id[nid].add("return")
            if isinstance(stmt, ast.Raise):
                self.roles_by_id[nid].add("raise")
            if isinstance(stmt, ast.Expr) and self._is_call_expr(stmt):
                self.roles_by_id[nid].add("call_stmt")

            next_controls = active_controls
            if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)):
                next_controls = active_controls + [nid]

            if hasattr(stmt, "body") and isinstance(stmt.body, list):
                self._walk_body_with_control(stmt.body, next_controls)
            if hasattr(stmt, "orelse") and isinstance(stmt.orelse, list):
                self._walk_body_with_control(stmt.orelse, next_controls)
            if isinstance(stmt, ast.Try):
                for h in stmt.handlers:
                    self._walk_body_with_control(h.body, next_controls)
                self._walk_body_with_control(stmt.finalbody, next_controls)

    def _build_data_dependencies(self) -> None:
        for nid, node in self.nodes_by_id.items():
            used_names = self._used_names(node)
            line_no = self.line_by_id.get(nid, -1)
            for name in sorted(used_names):
                defs = self.latest_def_by_name.get(name, [])
                dep = self._latest_def_before(defs, line_no, exclude_id=nid)
                if dep:
                    self.depends_on[nid].add(dep)
                    self.roles_by_id[nid].add("data_dependent")

    # -------------------------
    # heuristics
    # -------------------------

    def _is_output_node(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Return):
            return True
        if isinstance(node, ast.Raise):
            return True
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = _python_call_name(node.value)
            if name in {"print", "logger.debug", "logger.info", "logger.warning", "logger.error", "log"}:
                return True
            if _looks_like_python_file_write(node.value):
                return True
        return False

    def _looks_like_validation(self, node: ast.AST) -> bool:
        # Axiom 3 / EC4 approximation:
        # null checks, boundary guards, defensive early checks
        if isinstance(node, ast.If):
            text = self._code_of(node).lower()
            patterns = [
                " is none",
                " is not none",
                " not ",
                "len(",
                "== 0",
                "< 0",
                "<=",
                ">=",
                "assert",
                "raise",
            ]
            return any(p in text for p in patterns)
        if isinstance(node, ast.Assert):
            return True
        return False

    # -------------------------
    # utils
    # -------------------------

    def _node_id(self, node: ast.AST) -> str:
        return f"py:{getattr(self.fn, 'name', '<fn>')}:{getattr(node, 'lineno', -1)}:{type(node).__name__}"

    def _code_of(self, node: ast.AST) -> str:
        try:
            return ast.get_source_segment("\n".join(self.source_lines), node) or self._line_fallback(node)
        except Exception:
            return self._line_fallback(node)

    def _line_fallback(self, node: ast.AST) -> str:
        lineno = getattr(node, "lineno", None)
        if lineno is None or lineno <= 0 or lineno > len(self.source_lines):
            return type(node).__name__
        return self.source_lines[lineno - 1].rstrip()

    def _is_statement_like(self, node: ast.AST) -> bool:
        return isinstance(
            node,
            (
                ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Raise,
                ast.Expr, ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try,
                ast.With, ast.AsyncWith, ast.Assert,
            ),
        )

    def _is_definition_stmt(self, node: ast.AST) -> bool:
        return isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign))

    def _defined_names(self, node: ast.AST) -> Set[str]:
        names: Set[str] = set()
        targets = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]

        for t in targets:
            for child in ast.walk(t):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    names.add(child.id)
        return names

    def _used_names(self, node: ast.AST) -> Set[str]:
        names: Set[str] = set()
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                names.add(child.id)
        return names

    def _latest_def_before(
        self,
        defs: List[Tuple[int, str]],
        line_no: int,
        exclude_id: str,
    ) -> Optional[str]:
        best: Optional[str] = None
        best_line = -1
        for d_line, d_id in defs:
            if d_id == exclude_id:
                continue
            if d_line < line_no and d_line > best_line:
                best_line = d_line
                best = d_id
        return best

    def _is_call_expr(self, stmt: ast.Expr) -> bool:
        return isinstance(stmt.value, ast.Call)


def _python_signature_text(fn: ast.AST, lines: List[str]) -> str:
    lineno = getattr(fn, "lineno", -1)
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].rstrip()
    return getattr(fn, "name", "<function>")


def _python_call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        parts = []
        cur = call.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return "<call>"


def _looks_like_python_file_write(call: ast.Call) -> bool:
    name = _python_call_name(call)
    return name.endswith(".write") or name.endswith(".writelines")


# ============================================================
# Java local logic
# ============================================================

def _build_local_java(
    source: str,
    class_level: str,
    target_name: Optional[str],
    target_signature: Optional[str],
) -> Tuple[List[LocalFunctionBeacon], Dict[str, Any], List[str]]:
    """
    Java local rules are heuristic, text-based, lightweight.

    We do NOT attempt full Java parsing here.
    We extract method blocks and apply local dependency approximation on lines.
    """
    warnings: List[str] = []

    if not source.strip() and not class_level.strip():
        return [], {"java_parse": "empty_source"}, ["empty java source"]

    methods = _extract_java_method_blocks(source)
    results: List[LocalFunctionBeacon] = []

    if not methods and class_level.strip():
        warnings.append("no java method block extracted from source")

    for method_name, signature, start_line, block_lines in methods:
        if target_name and method_name != target_name:
            continue

        if target_name and method_name == target_name and target_signature:
            signature = target_signature

        index = _JavaMethodIndex(
            method_name=method_name,
            signature=signature,
            start_line=start_line,
            block_lines=block_lines,
        )
        index.build()

        output_ids = index.find_output_nodes()
        beacon_ids = index.backward_closure(output_ids)
        filtered_ids = index.filter_validation_nodes(beacon_ids, output_ids)
        reduced_ids = index.reduce_nodes(filtered_ids)

        nodes = [index.to_beacon_node(nid).to_dict() for nid in reduced_ids]

        results.append(
            LocalFunctionBeacon(
                function_name=method_name,
                signature=signature,
                lang="java",
                output_node_ids=sorted(output_ids),
                beacon_node_ids=sorted(reduced_ids),
                filtered_validation_node_ids=sorted(set(beacon_ids) - set(filtered_ids)),
                nodes=nodes,
                warnings=index.warnings.copy(),
            )
        )

    debug = {
        "method_count": len(methods),
        "selected_count": len(results),
    }
    return results, debug, warnings


class _JavaMethodIndex:
    """
    Lightweight line-based local dependence extractor for Java.

    Statement unit:
    - one normalized line in method block

    Dependence approximation:
    - variable definition/use backward linking
    - control blocks linked conservatively
    """

    def __init__(self, method_name: str, signature: str, start_line: int, block_lines: List[str]) -> None:
        self.method_name = method_name
        self.signature = signature
        self.start_line = start_line
        self.block_lines = block_lines

        self.code_by_id: Dict[str, str] = {}
        self.line_by_id: Dict[str, int] = {}
        self.kind_by_id: Dict[str, str] = {}
        self.roles_by_id: Dict[str, Set[str]] = {}
        self.depends_on: Dict[str, Set[str]] = {}
        self.latest_def_by_name: Dict[str, List[Tuple[int, str]]] = {}
        self.warnings: List[str] = []

    def build(self) -> None:
        for idx, raw in enumerate(self.block_lines, start=0):
            code = raw.rstrip()
            if not code.strip():
                continue
            abs_line = self.start_line + idx
            nid = self._node_id(abs_line)
            self.code_by_id[nid] = code
            self.line_by_id[nid] = abs_line
            self.kind_by_id[nid] = self._kind_of(code)
            self.roles_by_id[nid] = set()
            self.depends_on[nid] = set()

            defs = _java_defined_names(code)
            if defs:
                self.roles_by_id[nid].add("definition")
                for name in defs:
                    self.latest_def_by_name.setdefault(name, []).append((abs_line, nid))

        # data dependencies
        for nid, code in self.code_by_id.items():
            used = _java_used_names(code)
            line_no = self.line_by_id[nid]
            for name in sorted(used):
                defs = self.latest_def_by_name.get(name, [])
                dep = self._latest_def_before(defs, line_no, exclude_id=nid)
                if dep:
                    self.depends_on[nid].add(dep)
                    self.roles_by_id[nid].add("data_dependent")

        # conservative control dependency: if/for/while/try line affects following executable lines
        control_stack: List[str] = []
        brace_depth = 0
        control_depth_entries: List[Tuple[int, str]] = []

        for nid in sorted(self.code_by_id.keys(), key=lambda x: self.line_by_id[x]):
            code = self.code_by_id[nid].strip()

            # clear exited controls
            current_depth = brace_depth
            control_depth_entries = [item for item in control_depth_entries if item[0] <= current_depth]
            for _, ctrl_id in control_depth_entries:
                if ctrl_id != nid:
                    self.depends_on[nid].add(ctrl_id)

            if _looks_like_java_control(code):
                self.roles_by_id[nid].add("control")
                control_depth_entries.append((brace_depth + 1, nid))

            brace_depth += code.count("{")
            brace_depth -= code.count("}")

    def find_output_nodes(self) -> Set[str]:
        outputs: Set[str] = set()
        for nid, code in self.code_by_id.items():
            s = code.strip()
            if s.startswith("return " ) or s == "return;" or s.startswith("throw "):
                outputs.add(nid)
                self.roles_by_id[nid].add("output")
                continue
            if "System.out.print" in s or "logger." in s:
                outputs.add(nid)
                self.roles_by_id[nid].add("output")
                continue
            if ".write(" in s or ".append(" in s:
                outputs.add(nid)
                self.roles_by_id[nid].add("output")
        return outputs

    def backward_closure(self, seed_ids: Set[str]) -> Set[str]:
        visited = set(seed_ids)
        worklist = list(seed_ids)

        while worklist:
            nid = worklist.pop()
            for dep in sorted(self.depends_on.get(nid, set())):
                if dep not in visited:
                    visited.add(dep)
                    worklist.append(dep)
        return visited

    def filter_validation_nodes(self, beacon_ids: Set[str], output_ids: Set[str]) -> Set[str]:
        kept = set(beacon_ids)
        for nid in list(beacon_ids):
            code = self.code_by_id[nid].strip().lower()
            if _looks_like_java_validation(code) and nid not in output_ids:
                kept.discard(nid)
                self.roles_by_id[nid].add("validation_filtered")
        return kept

    def reduce_nodes(self, beacon_ids: Set[str]) -> List[str]:
        ordered = sorted(beacon_ids, key=lambda x: (self.line_by_id.get(x, 10**9), x))
        reduced: List[str] = []
        seen_norm: Set[Tuple[int, str]] = set()

        for nid in ordered:
            code = self.code_by_id[nid].strip()
            norm = _normalize_code(code)
            key = (self.line_by_id.get(nid, -1), norm)
            if key in seen_norm:
                continue
            seen_norm.add(key)
            reduced.append(nid)
        return reduced

    def to_beacon_node(self, nid: str) -> LocalBeaconNode:
        return LocalBeaconNode(
            node_id=nid,
            function_name=self.method_name,
            line_no=self.line_by_id.get(nid, -1),
            kind=self.kind_by_id.get(nid, "unknown"),
            code=self.code_by_id.get(nid, ""),
            roles=sorted(self.roles_by_id.get(nid, set())),
            depends_on=sorted(self.depends_on.get(nid, set())),
        )

    def _node_id(self, abs_line: int) -> str:
        return f"java:{self.method_name}:{abs_line}"

    def _kind_of(self, code: str) -> str:
        s = code.strip()
        if s.startswith("return"):
            return "return"
        if s.startswith("throw"):
            return "throw"
        if s.startswith("if"):
            return "if"
        if s.startswith("for"):
            return "for"
        if s.startswith("while"):
            return "while"
        if s.startswith("try"):
            return "try"
        if "=" in s:
            return "assign"
        if "(" in s and ")" in s:
            return "call_or_expr"
        return "stmt"

    def _latest_def_before(
        self,
        defs: List[Tuple[int, str]],
        line_no: int,
        exclude_id: str,
    ) -> Optional[str]:
        best: Optional[str] = None
        best_line = -1
        for d_line, d_id in defs:
            if d_id == exclude_id:
                continue
            if d_line < line_no and d_line > best_line:
                best_line = d_line
                best = d_id
        return best


def _extract_java_method_blocks(source: str) -> List[Tuple[str, str, int, List[str]]]:
    """
    Very lightweight brace-based Java method extraction.
    Returns:
        [(method_name, signature_line, start_line, block_lines), ...]
    """
    lines = source.splitlines()
    results: List[Tuple[str, str, int, List[str]]] = []

    method_sig_re = re.compile(
        r'^\s*(?:public|protected|private|static|final|native|synchronized|abstract|default|strictfp|\s)+'
        r'.*?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*(?:throws\s+[^{]+)?\{?\s*$'
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        m = method_sig_re.match(line)
        if not m:
            i += 1
            continue

        method_name = m.group(1)
        signature = line.rstrip()

        # find opening brace
        brace_line = i
        while brace_line < len(lines) and "{" not in lines[brace_line]:
            brace_line += 1
        if brace_line >= len(lines):
            i += 1
            continue

        depth = 0
        block_start = brace_line
        block_end = brace_line

        for j in range(brace_line, len(lines)):
            depth += lines[j].count("{")
            depth -= lines[j].count("}")
            if depth == 0:
                block_end = j
                break

        block_lines = lines[block_start:block_end + 1]
        results.append((method_name, signature, block_start + 1, block_lines))
        i = block_end + 1

    return results


def _java_defined_names(code: str) -> Set[str]:
    out: Set[str] = set()
    # examples:
    # int x = ...
    # String name = ...
    # x = ...
    m = re.match(r'^\s*(?:[A-Za-z_<>\[\]]+\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*=', code)
    if m:
        out.add(m.group(1))
    m2 = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=', code)
    if m2:
        out.add(m2.group(1))
    return out


def _java_used_names(code: str) -> Set[str]:
    tokens = set(re.findall(r'\b[A-Za-z_][A-Za-z0-9_]*\b', code))
    blacklist = {
        "return", "throw", "if", "else", "for", "while", "try", "catch", "finally",
        "new", "null", "true", "false", "public", "private", "protected", "static",
        "final", "class", "interface", "enum", "void", "int", "long", "double",
        "float", "boolean", "char", "byte", "short", "String", "System", "out",
    }
    return {t for t in tokens if t not in blacklist}


def _looks_like_java_control(code: str) -> bool:
    s = code.strip()
    return s.startswith(("if ", "if(", "for ", "for(", "while ", "while(", "try", "switch"))


def _looks_like_java_validation(code: str) -> bool:
    patterns = [
        "== null", "!= null", "isempty()", "length == 0", "size() == 0",
        "< 0", "<=", ">=", "throw new illegalargumentexception",
        "throw new nullpointerexception", "assert",
    ]
    return any(p in code for p in patterns)


# ============================================================
# Shared helpers
# ============================================================

def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}


def _read_text(data: Dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    return value if isinstance(value, str) else str(value or "")


def _read_optional_text(data: Dict[str, Any], key: str) -> Optional[str]:
    value = data.get(key)
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _normalize_code(code: str) -> str:
    return re.sub(r"\s+", " ", code.strip())