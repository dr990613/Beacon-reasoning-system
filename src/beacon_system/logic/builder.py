# src/beacon_system/logic/builder.py
# -*- coding: utf-8 -*-

"""
Build Raw Beacon IR from preprocess + local rules + global rules.

Design goals:
- keep internal structure rich enough for system use
- preserve nodes / edges / provenance / rule tags
- do NOT optimize for direct LLM readability here
- stable and schema-light output for tree/signatures downstream
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class RawIRNode:
    function_name: str
    line_no: int
    code: str
    kind: str
    roles: List[str]
    source: str  # local / global / synthetic

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RawIREdge:
    src_function: str
    src_line_no: int
    dst_function: str
    dst_line_no: int
    edge_type: str  # depends_on / call / return_flow / global_state
    label: str
    rule: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RawBeaconIR:
    lang: str
    entry_functions: List[str]
    functions: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    provenance: Dict[str, Any]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_raw_ir(preprocessed: Any, local_result: Any, global_result: Any) -> Dict[str, Any]:
    pre = _as_dict(preprocessed)
    loc = _as_dict(local_result)
    glob = _as_dict(global_result)

    lang = str(pre.get("lang", "unknown")).lower().strip()
    entry_functions = list(glob.get("entry_functions", []) or [])

    local_functions = _function_map(loc.get("functions", []) or [])
    global_functions = _function_map(glob.get("functions", []) or [])

    nodes: List[RawIRNode] = []
    edges: List[RawIREdge] = []
    node_keys: Set[Tuple[str, int, str]] = set()

    # 1. local/global nodes
    for fn_name, g_info in global_functions.items():
        local_node_ids = set(g_info.get("local_beacon_node_ids", []) or [])
        imported_ids = set(g_info.get("global_beacon_node_ids", []) or []) - local_node_ids

        l_info = local_functions.get(fn_name, {})
        local_nodes = l_info.get("nodes", []) or []

        for n in local_nodes:
            key = (fn_name, int(n.get("line_no", -1)), str(n.get("code", "")))
            if key in node_keys:
                continue
            node_keys.add(key)

            source = "local"
            if str(n.get("node_id", "")) in imported_ids:
                source = "global"

            nodes.append(
                RawIRNode(
                    function_name=fn_name,
                    line_no=int(n.get("line_no", -1)),
                    code=str(n.get("code", "")),
                    kind=str(n.get("kind", "unknown")),
                    roles=list(n.get("roles", []) or []),
                    source=source,
                )
            )

        # local depends_on edges
        by_line = {
            int(n.get("line_no", -1)): n
            for n in local_nodes
        }
        for n in local_nodes:
            dst_line = int(n.get("line_no", -1))
            for dep_id in n.get("depends_on", []) or []:
                dep_line = _extract_line_from_node_id(dep_id)
                if dep_line is None:
                    continue
                if dep_line not in by_line:
                    continue
                edges.append(
                    RawIREdge(
                        src_function=fn_name,
                        src_line_no=dep_line,
                        dst_function=fn_name,
                        dst_line_no=dst_line,
                        edge_type="depends_on",
                        label="local dependency",
                        rule="L-DEP",
                    )
                )

    # 2. call edges
    for e in glob.get("call_edges", []) or []:
        caller = str(e.get("caller", ""))
        callee = str(e.get("callee", ""))
        line_no = int(e.get("call_line_no", -1))
        edges.append(
            RawIREdge(
                src_function=caller,
                src_line_no=line_no,
                dst_function=callee,
                dst_line_no=_function_root_line(local_functions.get(callee, {})),
                edge_type="call",
                label="call propagation",
                rule=str(e.get("via_rule", "G-CALL")),
            )
        )

    # 3. return-flow edges
    for e in glob.get("ret_edges", []) or []:
        callee = str(e.get("callee", ""))
        caller = str(e.get("caller", ""))
        line_no = int(e.get("caller_line_no", -1))
        edges.append(
            RawIREdge(
                src_function=callee,
                src_line_no=_function_return_line(local_functions.get(callee, {})),
                dst_function=caller,
                dst_line_no=line_no,
                edge_type="return_flow",
                label="return-flow propagation",
                rule=str(e.get("via_rule", "G-RET")),
            )
        )

    # 4. global-state edges
    for e in glob.get("global_state_edges", []) or []:
        src_fn = str(e.get("src_function", ""))
        dst_fn = str(e.get("dst_function", ""))
        src_line = _extract_line_from_node_id(str(e.get("src_node_id", ""))) or -1
        dst_line = _extract_line_from_node_id(str(e.get("dst_node_id", ""))) or -1
        symbol = str(e.get("symbol", ""))
        edges.append(
            RawIREdge(
                src_function=src_fn,
                src_line_no=src_line,
                dst_function=dst_fn,
                dst_line_no=dst_line,
                edge_type="global_state",
                label=f"global state: {symbol}",
                rule=str(e.get("via_rule", "G-GLOB")),
            )
        )

    functions = []
    all_function_names = sorted(set(local_functions.keys()) | set(global_functions.keys()))
    for fn_name in all_function_names:
        l_info = local_functions.get(fn_name, {})
        g_info = global_functions.get(fn_name, {})
        functions.append(
            {
                "function_name": fn_name,
                "signature": l_info.get("signature") or g_info.get("signature"),
                "lang": lang,
                "local_beacon_node_ids": list(g_info.get("local_beacon_node_ids", []) or []),
                "global_beacon_node_ids": list(g_info.get("global_beacon_node_ids", []) or []),
                "output_node_ids": list(l_info.get("output_node_ids", []) or []),
                "nodes": list(l_info.get("nodes", []) or []),
            }
        )

    ir = RawBeaconIR(
        lang=lang,
        entry_functions=entry_functions,
        functions=functions,
        nodes=[n.to_dict() for n in _sort_nodes(nodes)],
        edges=[e.to_dict() for e in _sort_edges(edges)],
        provenance={
            "local_warnings": list(loc.get("warnings", []) or []),
            "global_warnings": list(glob.get("warnings", []) or []),
            "preprocess_warnings": list(pre.get("warnings", []) or []),
            "rules_used": ["L-DEP", "G-CALL", "G-RET", "G-GLOB"],
        },
        debug={
            "node_count": len(nodes),
            "edge_count": len(edges),
            "function_count": len(functions),
        },
    )
    return ir.to_dict()


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}


def _function_map(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for item in items:
        name = item.get("function_name")
        if name:
            out[name] = item
    return out


def _extract_line_from_node_id(node_id: str) -> Optional[int]:
    parts = node_id.split(":")
    if len(parts) < 3:
        return None
    try:
        return int(parts[2])
    except Exception:
        return None


def _function_root_line(info: Dict[str, Any]) -> int:
    nodes = info.get("nodes", []) or []
    if not nodes:
        return -1
    return min(int(n.get("line_no", 10**9)) for n in nodes)


def _function_return_line(info: Dict[str, Any]) -> int:
    nodes = info.get("nodes", []) or []
    returns = [int(n.get("line_no", -1)) for n in nodes if "return" in list(n.get("roles", []) or [])]
    if returns:
        return min(returns)
    return _function_root_line(info)


def _sort_nodes(nodes: List[RawIRNode]) -> List[RawIRNode]:
    return sorted(nodes, key=lambda x: (x.function_name, x.line_no, x.code))


def _sort_edges(edges: List[RawIREdge]) -> List[RawIREdge]:
    return sorted(
        edges,
        key=lambda x: (
            x.src_function,
            x.src_line_no,
            x.dst_function,
            x.dst_line_no,
            x.edge_type,
        ),
    )