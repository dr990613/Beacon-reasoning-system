# src/beacon_system/logic/rules_global.py
# -*- coding: utf-8 -*-

"""
Global Beacon Logic rules.

Responsibilities:
- lift local beacons to global beacons
- detect entry function(s)
- propagate beacons across calls
- approximate return-flow propagation
- conservatively handle global state
- produce program-level beacon relations

Non-goals:
- no code generation
- no verifier behavior
- no heavy whole-program semantic analysis
- no planner-like reasoning

Expected inputs:
1) preprocessed: result from logic.preprocess.preprocess_task(...)
2) local_result: result from logic.rules_local.build_local_beacons(...)

Output:
- stable dict for downstream builder/tree/signatures
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
class GlobalCallEdge:
    caller: str
    callee: str
    call_node_id: str
    call_line_no: int
    call_code: str
    via_rule: str = "G-CALL"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalRetEdge:
    callee: str
    caller: str
    caller_node_id: str
    caller_line_no: int
    caller_code: str
    via_rule: str = "G-RET"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalStateEdge:
    symbol: str
    src_function: str
    dst_function: str
    src_node_id: str
    dst_node_id: str
    via_rule: str = "G-GLOB"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalFunctionBeacon:
    function_name: str
    signature: Optional[str]
    lang: str
    local_beacon_node_ids: List[str]
    global_beacon_node_ids: List[str]
    imported_from_calls: List[str]
    imported_from_returns: List[str]
    imported_from_globals: List[str]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class GlobalRulesResult:
    lang: str
    entry_functions: List[str]
    functions: List[Dict[str, Any]]
    call_edges: List[Dict[str, Any]]
    ret_edges: List[Dict[str, Any]]
    global_state_edges: List[Dict[str, Any]]
    program_beacon_node_ids: List[str]
    warnings: List[str]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Public API
# ============================================================

def build_global_beacons(preprocessed: Any, local_result: Any) -> Dict[str, Any]:
    data = _as_dict(preprocessed)
    loc = _as_dict(local_result)

    lang = str(data.get("lang", "unknown")).lower().strip()
    source = _read_text(data, "body_text") or _read_text(data, "file_content_text") or _read_text(data, "class_level_text")
    class_level = _read_text(data, "class_level_text")
    local_functions = loc.get("functions", []) or []

    warnings: List[str] = []

    if lang == "python":
        result = _build_global_python(
            source=source,
            class_level=class_level,
            local_functions=local_functions,
        )
    elif lang == "java":
        result = _build_global_java(
            source=source,
            class_level=class_level,
            local_functions=local_functions,
        )
    else:
        result = GlobalRulesResult(
            lang=lang,
            entry_functions=[],
            functions=[],
            call_edges=[],
            ret_edges=[],
            global_state_edges=[],
            program_beacon_node_ids=[],
            warnings=[f"unsupported language '{lang}' in global rules"],
            debug={"reason": "unsupported language"},
        )
    return result.to_dict()


# ============================================================
# Python global logic
# ============================================================

def _build_global_python(
    source: str,
    class_level: str,
    local_functions: List[Dict[str, Any]],
) -> GlobalRulesResult:
    warnings: List[str] = []

    if not source.strip():
        return GlobalRulesResult(
            lang="python",
            entry_functions=[],
            functions=[],
            call_edges=[],
            ret_edges=[],
            global_state_edges=[],
            program_beacon_node_ids=[],
            warnings=["empty python source"],
            debug={"python_parse": "empty_source"},
        )

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return GlobalRulesResult(
            lang="python",
            entry_functions=[],
            functions=[],
            call_edges=[],
            ret_edges=[],
            global_state_edges=[],
            program_beacon_node_ids=[],
            warnings=["python source parse failed in global rules"],
            debug={"python_parse": "syntax_error", "detail": str(exc)},
        )

    local_map = _local_function_map(local_functions)
    fn_nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    fn_names = [n.name for n in fn_nodes]
    defined_functions = set(fn_names)
    entry_functions = _infer_python_entries(fn_nodes, local_map)

    py_index = _PythonGlobalIndex(tree=tree, source=source, local_map=local_map)
    py_index.build()

    _run_global_fixpoint(
        function_names=list(local_map.keys()),
        global_sets=py_index.global_sets,
        call_relations=py_index.call_relations,
        ret_relations=py_index.ret_relations,
        read_write_relations=py_index.global_symbol_relations,
    )

    call_edges = [
        GlobalCallEdge(
            caller=caller,
            callee=callee,
            call_node_id=node_id,
            call_line_no=line_no,
            call_code=code,
        ).to_dict()
        for caller, callee, node_id, line_no, code in py_index.materialized_call_edges()
    ]

    ret_edges = [
        GlobalRetEdge(
            callee=callee,
            caller=caller,
            caller_node_id=node_id,
            caller_line_no=line_no,
            caller_code=code,
        ).to_dict()
        for callee, caller, node_id, line_no, code in py_index.materialized_ret_edges()
    ]

    global_state_edges = [
        GlobalStateEdge(
            symbol=symbol,
            src_function=src_fn,
            dst_function=dst_fn,
            src_node_id=src_node,
            dst_node_id=dst_node,
        ).to_dict()
        for symbol, src_fn, dst_fn, src_node, dst_node in py_index.materialized_global_edges()
    ]

    function_results = []
    for fn_name, local_info in local_map.items():
        global_nodes = sorted(py_index.global_sets.get(fn_name, set()))
        local_nodes = sorted(set(local_info.get("beacon_node_ids", []) or []))

        imported_from_calls = sorted(set(global_nodes) - set(local_nodes))
        imported_from_returns = sorted(py_index.imported_by_ret.get(fn_name, set()))
        imported_from_globals = sorted(py_index.imported_by_global.get(fn_name, set()))

        function_results.append(
            GlobalFunctionBeacon(
                function_name=fn_name,
                signature=local_info.get("signature"),
                lang="python",
                local_beacon_node_ids=local_nodes,
                global_beacon_node_ids=global_nodes,
                imported_from_calls=imported_from_calls,
                imported_from_returns=imported_from_returns,
                imported_from_globals=imported_from_globals,
                warnings=[],
            ).to_dict()
        )

    program_beacon_node_ids = sorted({
        nid
        for entry in entry_functions
        for nid in py_index.global_sets.get(entry, set())
    })

    return GlobalRulesResult(
        lang="python",
        entry_functions=entry_functions,
        functions=function_results,
        call_edges=call_edges,
        ret_edges=ret_edges,
        global_state_edges=global_state_edges,
        program_beacon_node_ids=program_beacon_node_ids,
        warnings=warnings,
        debug={
            "function_count": len(fn_nodes),
            "local_function_count": len(local_map),
            "entry_count": len(entry_functions),
            "defined_functions": sorted(defined_functions),
        },
    )


class _PythonGlobalIndex:
    def __init__(self, tree: ast.AST, source: str, local_map: Dict[str, Dict[str, Any]]) -> None:
        self.tree = tree
        self.source = source
        self.lines = source.splitlines()
        self.local_map = local_map

        self.fn_nodes: Dict[str, ast.AST] = {}
        self.global_sets: Dict[str, Set[str]] = {}
        self.call_relations: Dict[str, List[Tuple[str, str, int, str]]] = {}
        self.ret_relations: Dict[str, List[Tuple[str, int, str]]] = {}
        self.global_symbol_relations: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        self.imported_by_ret: Dict[str, Set[str]] = {}
        self.imported_by_global: Dict[str, Set[str]] = {}

    def build(self) -> None:
        self._index_functions()
        self._seed_local_to_global()
        self._index_calls()
        self._index_return_flow()
        self._index_global_symbols()

    def _index_functions(self) -> None:
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.fn_nodes[node.name] = node

    def _seed_local_to_global(self) -> None:
        for fn_name, info in self.local_map.items():
            self.global_sets[fn_name] = set(info.get("beacon_node_ids", []) or [])
            self.call_relations[fn_name] = []
            self.ret_relations[fn_name] = []
            self.imported_by_ret[fn_name] = set()
            self.imported_by_global[fn_name] = set()

    def _index_calls(self) -> None:
        for fn_name, info in self.local_map.items():
            node = self.fn_nodes.get(fn_name)
            if node is None:
                continue

            local_ids = set(info.get("beacon_node_ids", []) or [])
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    callee = _python_call_name(child)
                    if callee not in self.local_map:
                        continue

                    stmt = _nearest_python_stmt(node, child)
                    if stmt is None:
                        continue
                    stmt_id = _python_stmt_node_id(fn_name, stmt)
                    if stmt_id not in local_ids:
                        continue

                    line_no = getattr(stmt, "lineno", -1)
                    code = _safe_source_segment(self.source, stmt)
                    self.call_relations[fn_name].append((callee, stmt_id, line_no, code))

    def _index_return_flow(self) -> None:
        """
        Heuristic G-RET:
        if a beacon statement in caller assigns from callee(...) or returns callee(...),
        then import callee global beacons to caller.
        """
        for fn_name, info in self.local_map.items():
            node = self.fn_nodes.get(fn_name)
            if node is None:
                continue

            local_ids = set(info.get("beacon_node_ids", []) or [])
            for child in ast.walk(node):
                if isinstance(child, (ast.Assign, ast.AnnAssign, ast.Return)):
                    stmt_id = _python_stmt_node_id(fn_name, child)
                    if stmt_id not in local_ids:
                        continue

                    calls = _python_calls_in_stmt(child)
                    for call in calls:
                        callee = _python_call_name(call)
                        if callee not in self.local_map:
                            continue
                        line_no = getattr(child, "lineno", -1)
                        code = _safe_source_segment(self.source, child)
                        self.ret_relations[fn_name].append((callee, line_no, code))

    def _index_global_symbols(self) -> None:
        module_globals = _python_module_globals(self.tree)
        if not module_globals:
            return

        reads: Dict[str, List[Tuple[str, str]]] = {g: [] for g in module_globals}
        writes: Dict[str, List[Tuple[str, str]]] = {g: [] for g in module_globals}

        for fn_name, node in self.fn_nodes.items():
            for child in ast.walk(node):
                stmt = _nearest_python_stmt(node, child)
                if stmt is None:
                    continue
                stmt_id = _python_stmt_node_id(fn_name, stmt)

                if isinstance(child, ast.Name):
                    if child.id not in module_globals:
                        continue
                    if isinstance(child.ctx, ast.Load):
                        reads[child.id].append((fn_name, stmt_id))
                    elif isinstance(child.ctx, (ast.Store, ast.Del)):
                        writes[child.id].append((fn_name, stmt_id))

        for sym in module_globals:
            self.global_symbol_relations[sym] = {
                "reads": reads.get(sym, []),
                "writes": writes.get(sym, []),
            }

    def materialized_call_edges(self) -> List[Tuple[str, str, str, int, str]]:
        out = []
        for caller, rels in self.call_relations.items():
            for callee, stmt_id, line_no, code in rels:
                if stmt_id in self.global_sets.get(caller, set()):
                    out.append((caller, callee, stmt_id, line_no, code))
        return out

    def materialized_ret_edges(self) -> List[Tuple[str, str, str, int, str]]:
        out = []
        for caller, rels in self.ret_relations.items():
            for callee, line_no, code in rels:
                synthetic_node = f"py:{caller}:{line_no}:retflow"
                if any(n in self.global_sets.get(caller, set()) for n in self.global_sets.get(callee, set())):
                    out.append((callee, caller, synthetic_node, line_no, code))
                    self.imported_by_ret[caller].update(self.global_sets.get(callee, set()))
        return out

    def materialized_global_edges(self) -> List[Tuple[str, str, str, str, str]]:
        out = []
        for sym, rels in self.global_symbol_relations.items():
            reads = rels.get("reads", [])
            writes = rels.get("writes", [])
            if not reads or not writes:
                continue

            read_hit = any(
                node_id in self.global_sets.get(fn_name, set())
                for fn_name, node_id in reads
            )
            write_hit = any(
                node_id in self.global_sets.get(fn_name, set())
                for fn_name, node_id in writes
            )
            if not (read_hit or write_hit):
                continue

            for src_fn, src_node in writes:
                for dst_fn, dst_node in reads:
                    out.append((sym, src_fn, dst_fn, src_node, dst_node))
                    self.imported_by_global[dst_fn].add(src_node)
                    self.imported_by_global[src_fn].add(dst_node)
        return out


# ============================================================
# Java global logic
# ============================================================

def _build_global_java(
    source: str,
    class_level: str,
    local_functions: List[Dict[str, Any]],
) -> GlobalRulesResult:
    warnings: List[str] = []
    local_map = _local_function_map(local_functions)
    entry_functions = _infer_java_entries(source, local_map)

    j_index = _JavaGlobalIndex(
        source=source,
        class_level=class_level,
        local_map=local_map,
    )
    j_index.build()

    _run_global_fixpoint(
        function_names=list(local_map.keys()),
        global_sets=j_index.global_sets,
        call_relations=j_index.call_relations,
        ret_relations=j_index.ret_relations,
        read_write_relations=j_index.global_symbol_relations,
    )

    call_edges = [
        GlobalCallEdge(
            caller=caller,
            callee=callee,
            call_node_id=node_id,
            call_line_no=line_no,
            call_code=code,
        ).to_dict()
        for caller, callee, node_id, line_no, code in j_index.materialized_call_edges()
    ]

    ret_edges = [
        GlobalRetEdge(
            callee=callee,
            caller=caller,
            caller_node_id=node_id,
            caller_line_no=line_no,
            caller_code=code,
        ).to_dict()
        for callee, caller, node_id, line_no, code in j_index.materialized_ret_edges()
    ]

    global_state_edges = [
        GlobalStateEdge(
            symbol=symbol,
            src_function=src_fn,
            dst_function=dst_fn,
            src_node_id=src_node,
            dst_node_id=dst_node,
        ).to_dict()
        for symbol, src_fn, dst_fn, src_node, dst_node in j_index.materialized_global_edges()
    ]

    function_results = []
    for fn_name, local_info in local_map.items():
        global_nodes = sorted(j_index.global_sets.get(fn_name, set()))
        local_nodes = sorted(set(local_info.get("beacon_node_ids", []) or []))
        imported_from_calls = sorted(set(global_nodes) - set(local_nodes))
        imported_from_returns = sorted(j_index.imported_by_ret.get(fn_name, set()))
        imported_from_globals = sorted(j_index.imported_by_global.get(fn_name, set()))

        function_results.append(
            GlobalFunctionBeacon(
                function_name=fn_name,
                signature=local_info.get("signature"),
                lang="java",
                local_beacon_node_ids=local_nodes,
                global_beacon_node_ids=global_nodes,
                imported_from_calls=imported_from_calls,
                imported_from_returns=imported_from_returns,
                imported_from_globals=imported_from_globals,
                warnings=[],
            ).to_dict()
        )

    program_beacon_node_ids = sorted({
        nid
        for entry in entry_functions
        for nid in j_index.global_sets.get(entry, set())
    })

    return GlobalRulesResult(
        lang="java",
        entry_functions=entry_functions,
        functions=function_results,
        call_edges=call_edges,
        ret_edges=ret_edges,
        global_state_edges=global_state_edges,
        program_beacon_node_ids=program_beacon_node_ids,
        warnings=warnings,
        debug={
            "local_function_count": len(local_map),
            "entry_count": len(entry_functions),
        },
    )


class _JavaGlobalIndex:
    def __init__(self, source: str, class_level: str, local_map: Dict[str, Dict[str, Any]]) -> None:
        self.source = source
        self.class_level = class_level
        self.local_map = local_map

        self.methods = _extract_java_method_blocks(source)
        self.global_sets: Dict[str, Set[str]] = {}
        self.call_relations: Dict[str, List[Tuple[str, str, int, str]]] = {}
        self.ret_relations: Dict[str, List[Tuple[str, int, str]]] = {}
        self.global_symbol_relations: Dict[str, Dict[str, List[Tuple[str, str]]]] = {}
        self.imported_by_ret: Dict[str, Set[str]] = {}
        self.imported_by_global: Dict[str, Set[str]] = {}

    def build(self) -> None:
        self._seed_local_to_global()
        self._index_calls()
        self._index_return_flow()
        self._index_global_symbols()

    def _seed_local_to_global(self) -> None:
        for fn_name, info in self.local_map.items():
            self.global_sets[fn_name] = set(info.get("beacon_node_ids", []) or [])
            self.call_relations[fn_name] = []
            self.ret_relations[fn_name] = []
            self.imported_by_ret[fn_name] = set()
            self.imported_by_global[fn_name] = set()

    def _index_calls(self) -> None:
        for method_name, signature, start_line, block_lines in self.methods:
            if method_name not in self.local_map:
                continue
            local_ids = set(self.local_map[method_name].get("beacon_node_ids", []) or [])

            for offset, raw in enumerate(block_lines):
                line_no = start_line + offset
                code = raw.rstrip()
                node_id = f"java:{method_name}:{line_no}"
                if node_id not in local_ids:
                    continue

                callees = _java_called_function_names(code)
                for callee in callees:
                    if callee in self.local_map and callee != method_name:
                        self.call_relations[method_name].append((callee, node_id, line_no, code))

    def _index_return_flow(self) -> None:
        for method_name, signature, start_line, block_lines in self.methods:
            if method_name not in self.local_map:
                continue
            local_ids = set(self.local_map[method_name].get("beacon_node_ids", []) or [])

            for offset, raw in enumerate(block_lines):
                line_no = start_line + offset
                code = raw.rstrip()
                node_id = f"java:{method_name}:{line_no}"
                if node_id not in local_ids:
                    continue

                callees = _java_called_function_names(code)
                if "=" in code or "return " in code:
                    for callee in callees:
                        if callee in self.local_map and callee != method_name:
                            self.ret_relations[method_name].append((callee, line_no, code))

    def _index_global_symbols(self) -> None:
        global_fields = _java_class_fields(self.class_level)
        if not global_fields:
            return

        reads: Dict[str, List[Tuple[str, str]]] = {g: [] for g in global_fields}
        writes: Dict[str, List[Tuple[str, str]]] = {g: [] for g in global_fields}

        for method_name, signature, start_line, block_lines in self.methods:
            if method_name not in self.local_map:
                continue
            for offset, raw in enumerate(block_lines):
                line_no = start_line + offset
                code = raw.rstrip()
                node_id = f"java:{method_name}:{line_no}"

                for field in global_fields:
                    if re.search(rf"\b{re.escape(field)}\b", code):
                        if _looks_like_java_write_to_symbol(code, field):
                            writes[field].append((method_name, node_id))
                        else:
                            reads[field].append((method_name, node_id))

        for sym in global_fields:
            self.global_symbol_relations[sym] = {
                "reads": reads.get(sym, []),
                "writes": writes.get(sym, []),
            }

    def materialized_call_edges(self) -> List[Tuple[str, str, str, int, str]]:
        out = []
        for caller, rels in self.call_relations.items():
            for callee, stmt_id, line_no, code in rels:
                if stmt_id in self.global_sets.get(caller, set()):
                    out.append((caller, callee, stmt_id, line_no, code))
        return out

    def materialized_ret_edges(self) -> List[Tuple[str, str, str, int, str]]:
        out = []
        for caller, rels in self.ret_relations.items():
            for callee, line_no, code in rels:
                synthetic_node = f"java:{caller}:{line_no}:retflow"
                if any(n in self.global_sets.get(caller, set()) for n in self.global_sets.get(callee, set())):
                    out.append((callee, caller, synthetic_node, line_no, code))
                    self.imported_by_ret[caller].update(self.global_sets.get(callee, set()))
        return out

    def materialized_global_edges(self) -> List[Tuple[str, str, str, str, str]]:
        out = []
        for sym, rels in self.global_symbol_relations.items():
            reads = rels.get("reads", [])
            writes = rels.get("writes", [])
            if not reads or not writes:
                continue

            read_hit = any(
                node_id in self.global_sets.get(fn_name, set())
                for fn_name, node_id in reads
            )
            write_hit = any(
                node_id in self.global_sets.get(fn_name, set())
                for fn_name, node_id in writes
            )
            if not (read_hit or write_hit):
                continue

            for src_fn, src_node in writes:
                for dst_fn, dst_node in reads:
                    out.append((sym, src_fn, dst_fn, src_node, dst_node))
                    self.imported_by_global[dst_fn].add(src_node)
                    self.imported_by_global[src_fn].add(dst_node)
        return out


# ============================================================
# Fixpoint engine
# ============================================================

def _run_global_fixpoint(
    function_names: List[str],
    global_sets: Dict[str, Set[str]],
    call_relations: Dict[str, List[Tuple[str, str, int, str]]],
    ret_relations: Dict[str, List[Tuple[str, int, str]]],
    read_write_relations: Dict[str, Dict[str, List[Tuple[str, str]]]],
) -> None:
    """
    Implements:
    - G-BASE   : local seeds already copied into global_sets
    - G-CALL   : if beacon call-site in caller, import callee global beacons
    - G-RET    : if return-flow relation exists for relevant beacon stmt, import callee global beacons
    - G-GLOB   : if any read/write of global symbol is beacon-relevant, conservatively connect all reads/writes
    """
    changed = True
    while changed:
        changed = False

        # G-CALL
        for caller in function_names:
            for callee, call_node_id, _line_no, _code in call_relations.get(caller, []):
                if call_node_id in global_sets.get(caller, set()):
                    before = len(global_sets[caller])
                    global_sets[caller].update(global_sets.get(callee, set()))
                    if len(global_sets[caller]) > before:
                        changed = True

        # G-RET
        for caller in function_names:
            for callee, _line_no, _code in ret_relations.get(caller, []):
                before = len(global_sets[caller])
                global_sets[caller].update(global_sets.get(callee, set()))
                if len(global_sets[caller]) > before:
                    changed = True

        # G-GLOB / G-GLOB-2
        for _sym, rels in read_write_relations.items():
            reads = rels.get("reads", [])
            writes = rels.get("writes", [])

            read_hit = any(
                node_id in global_sets.get(fn_name, set())
                for fn_name, node_id in reads
            )
            write_hit = any(
                node_id in global_sets.get(fn_name, set())
                for fn_name, node_id in writes
            )
            if not (read_hit or write_hit):
                continue

            full_cluster = reads + writes
            for dst_fn, _dst_node in full_cluster:
                before = len(global_sets[dst_fn])
                for src_fn, src_node in full_cluster:
                    global_sets[dst_fn].add(src_node)
                if len(global_sets[dst_fn]) > before:
                    changed = True


# ============================================================
# Entry inference
# ============================================================

def _infer_python_entries(fn_nodes: List[ast.AST], local_map: Dict[str, Dict[str, Any]]) -> List[str]:
    names = [n.name for n in fn_nodes]

    # EC9: prefer main, else public-API-like top-level functions :contentReference[oaicite:0]{index=0}
    if "main" in names and "main" in local_map:
        return ["main"]

    api_like = []
    for name in names:
        if name.startswith("_"):
            continue
        if name in local_map:
            api_like.append(name)

    return api_like[:3] if api_like else list(local_map.keys())[:1]


def _infer_java_entries(source: str, local_map: Dict[str, Dict[str, Any]]) -> List[str]:
    names = set(local_map.keys())

    if "main" in names:
        return ["main"]

    public_methods = []
    for method_name, signature, _start, _block in _extract_java_method_blocks(source):
        if method_name not in names:
            continue
        if "public " in signature:
            public_methods.append(method_name)

    return public_methods[:3] if public_methods else list(local_map.keys())[:1]


# ============================================================
# Shared helpers
# ============================================================

def _local_function_map(local_functions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for item in local_functions:
        name = item.get("function_name")
        if name:
            out[name] = item
    return out


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}


def _read_text(data: Dict[str, Any], key: str) -> str:
    value = data.get(key, "")
    return value if isinstance(value, str) else str(value or "")


# ---------------- Python helpers ----------------

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


def _python_stmt_node_id(function_name: str, stmt: ast.AST) -> str:
    return f"py:{function_name}:{getattr(stmt, 'lineno', -1)}:{type(stmt).__name__}"


def _nearest_python_stmt(fn_node: ast.AST, child: ast.AST) -> Optional[ast.AST]:
    best = None
    best_span = None
    for n in ast.walk(fn_node):
        if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Return, ast.Raise, ast.Expr, ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.Assert)):
            if not hasattr(n, "lineno") or not hasattr(child, "lineno"):
                continue
            if n.lineno <= child.lineno <= getattr(n, "end_lineno", n.lineno):
                span = getattr(n, "end_lineno", n.lineno) - n.lineno
                if best is None or span < best_span:
                    best = n
                    best_span = span
    return best


def _safe_source_segment(source: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(source, node) or type(node).__name__
    except Exception:
        return type(node).__name__


def _python_calls_in_stmt(stmt: ast.AST) -> List[ast.Call]:
    return [n for n in ast.walk(stmt) if isinstance(n, ast.Call)]


def _python_module_globals(tree: ast.AST) -> Set[str]:
    out = set()
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out.add(target.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                out.add(node.target.id)
    return out


# ---------------- Java helpers ----------------

def _extract_java_method_blocks(source: str) -> List[Tuple[str, str, int, List[str]]]:
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


def _java_called_function_names(code: str) -> List[str]:
    calls = re.findall(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(', code)
    blacklist = {
        "if", "for", "while", "switch", "catch", "return", "new", "throw",
        "super", "this", "synchronized",
    }
    return [c for c in calls if c not in blacklist]


def _java_class_fields(class_level: str) -> Set[str]:
    out = set()
    for line in class_level.splitlines():
        s = line.strip().rstrip(";")
        if not s or s.startswith("import "):
            continue
        m = re.match(r'^(?:[A-Za-z_<>\[\]]+\s+)+([A-Za-z_][A-Za-z0-9_]*)$', s)
        if m and "(" not in s:
            out.add(m.group(1))
    return out


def _looks_like_java_write_to_symbol(code: str, symbol: str) -> bool:
    return bool(re.search(rf'\b{re.escape(symbol)}\b\s*=', code))


# ============================================================
# end
# ============================================================