# src/beacon_system/logic/tree.py
# -*- coding: utf-8 -*-

"""
Project Raw Beacon IR to a human-readable Program Beacon Tree.

Design goals:
- output readable function-grouped tree
- show statement dependency chains
- show revisited nodes
- show key return paths
- avoid exposing raw internal ids to the final tree text
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set, Tuple


@dataclass
class TreeStatement:
    function_name: str
    line_no: int
    code: str
    children: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FunctionTree:
    function_name: str
    signature: Optional[str]
    root_statements: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BeaconTree:
    entry_function: Optional[str]
    functions: List[Dict[str, Any]]
    rendered_text: str
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_beacon_tree(raw_ir: Any) -> Dict[str, Any]:
    ir = _as_dict(raw_ir)

    entry_functions = list(ir.get("entry_functions", []) or [])
    entry_function = entry_functions[0] if entry_functions else None

    functions = ir.get("functions", []) or []
    nodes = ir.get("nodes", []) or []
    edges = ir.get("edges", []) or []

    nodes_by_fn = _group_nodes_by_function(nodes)
    dep_children = _build_dependency_children(edges)

    function_trees: List[FunctionTree] = []
    rendered_parts: List[str] = []

    header = f"Program Beacon Tree (Entry = {entry_function or 'unknown'})"
    rendered_parts.append(header)
    rendered_parts.append("─" * 42)

    ordered_functions = _ordered_functions(functions, entry_function)

    for idx, fn in enumerate(ordered_functions):
        fn_name = str(fn.get("function_name", "unknown"))
        signature = fn.get("signature")
        fn_nodes = nodes_by_fn.get(fn_name, [])

        root_nodes = _select_root_nodes(fn_nodes)
        rendered_parts.append(_render_function_header(fn_name, signature, idx == len(ordered_functions) - 1))

        tree_roots = []
        for i, root in enumerate(root_nodes):
            visited: Set[Tuple[str, int]] = set()
            stmt_tree = _build_stmt_tree(
                fn_name=fn_name,
                root=root,
                dep_children=dep_children,
                nodes_by_fn=nodes_by_fn,
                visited=visited,
            )
            tree_roots.append(stmt_tree.to_dict())
            rendered_parts.extend(
                _render_stmt_tree(
                    stmt_tree=stmt_tree,
                    prefix="   " if idx == len(ordered_functions) - 1 else "│  ",
                    is_last=(i == len(root_nodes) - 1),
                )
            )

        function_trees.append(
            FunctionTree(
                function_name=fn_name,
                signature=signature,
                root_statements=tree_roots,
            )
        )

    return BeaconTree(
        entry_function=entry_function,
        functions=[f.to_dict() for f in function_trees],
        rendered_text="\n".join(rendered_parts),
        debug={
            "function_count": len(function_trees),
            "entry_function": entry_function,
        },
    ).to_dict()


def _build_stmt_tree(
    fn_name: str,
    root: Dict[str, Any],
    dep_children: Dict[Tuple[str, int], List[Tuple[str, int]]],
    nodes_by_fn: Dict[str, List[Dict[str, Any]]],
    visited: Set[Tuple[str, int]],
) -> TreeStatement:
    key = (fn_name, int(root.get("line_no", -1)))
    visited.add(key)

    children: List[Dict[str, Any]] = []
    for child_key in dep_children.get(key, []):
        c_fn, c_line = child_key
        c_node = _find_node(nodes_by_fn.get(c_fn, []), c_line)
        if not c_node:
            continue

        if child_key in visited:
            children.append(
                {
                    "function_name": c_fn,
                    "line_no": c_line,
                    "code": "[visited]",
                    "children": [],
                    "visited_ref": True,
                    "visited_code": str(c_node.get("code", "")),
                }
            )
            continue

        subtree = _build_stmt_tree(
            fn_name=c_fn,
            root=c_node,
            dep_children=dep_children,
            nodes_by_fn=nodes_by_fn,
            visited=set(visited),
        )
        children.append(subtree.to_dict())

    return TreeStatement(
        function_name=fn_name,
        line_no=int(root.get("line_no", -1)),
        code=str(root.get("code", "")),
        children=children,
    )


def _render_stmt_tree(stmt_tree: TreeStatement, prefix: str, is_last: bool) -> List[str]:
    lines: List[str] = []
    branch = "└─ " if is_last else "├─ "
    line = f"{prefix}{branch}[{stmt_tree.function_name}] line {stmt_tree.line_no}:     {stmt_tree.code}"
    lines.append(line)

    child_prefix = prefix + ("   " if is_last else "│  ")
    total = len(stmt_tree.children)

    for i, child in enumerate(stmt_tree.children):
        child_is_last = (i == total - 1)

        if child.get("visited_ref"):
            branch2 = "└─ " if child_is_last else "├─ "
            lines.append(
                f"{child_prefix}{branch2}[visited] [{child['function_name']}] line {child['line_no']}:     {child['visited_code']}"
            )
            continue

        child_stmt = TreeStatement(
            function_name=child["function_name"],
            line_no=int(child["line_no"]),
            code=child["code"],
            children=child.get("children", []) or [],
        )
        lines.extend(_render_stmt_tree(child_stmt, child_prefix, child_is_last))

    return lines


def _render_function_header(fn_name: str, signature: Optional[str], is_last: bool) -> str:
    branch = "└─ " if is_last else "├─ "
    if signature:
        return f"{branch}Function {fn_name}{_signature_suffix(signature)}"
    return f"{branch}Function {fn_name}()"


def _signature_suffix(signature: str) -> str:
    sig = signature.strip()
    if sig.startswith("def ") or sig.startswith("async def "):
        start = sig.find("(")
        if start != -1:
            return sig[start:]
    if "(" in sig:
        return sig[sig.find("("):].rstrip("{").strip()
    return "()"


def _select_root_nodes(fn_nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # 优先 output 节点；没有则退回最晚的关键语句
    outputs = [n for n in fn_nodes if "output" in list(n.get("roles", []) or []) or "return" in list(n.get("roles", []) or [])]
    if outputs:
        return sorted(outputs, key=lambda x: int(x.get("line_no", -1)))
    return sorted(fn_nodes, key=lambda x: int(x.get("line_no", -1)))[-1:]


def _build_dependency_children(edges: List[Dict[str, Any]]) -> Dict[Tuple[str, int], List[Tuple[str, int]]]:
    out: Dict[Tuple[str, int], List[Tuple[str, int]]] = {}

    # 这里只把“依赖谁”投影成树形 child
    for e in edges:
        et = str(e.get("edge_type", ""))
        if et not in {"depends_on", "return_flow", "global_state", "call"}:
            continue

        src = (str(e.get("src_function", "")), int(e.get("src_line_no", -1)))
        dst = (str(e.get("dst_function", "")), int(e.get("dst_line_no", -1)))

        # 树里表现为 dst <- src
        out.setdefault(dst, [])
        if src not in out[dst]:
            out[dst].append(src)

    for k in list(out.keys()):
        out[k] = sorted(out[k], key=lambda x: (x[0], x[1]))
    return out


def _group_nodes_by_function(nodes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for n in nodes:
        fn = str(n.get("function_name", ""))
        out.setdefault(fn, []).append(n)
    for fn in out:
        out[fn] = sorted(out[fn], key=lambda x: int(x.get("line_no", -1)))
    return out


def _ordered_functions(functions: List[Dict[str, Any]], entry_function: Optional[str]) -> List[Dict[str, Any]]:
    if not entry_function:
        return sorted(functions, key=lambda x: str(x.get("function_name", "")))
    head = [f for f in functions if f.get("function_name") == entry_function]
    tail = sorted(
        [f for f in functions if f.get("function_name") != entry_function],
        key=lambda x: str(x.get("function_name", "")),
    )
    return tail + head if False else head + tail


def _find_node(nodes: List[Dict[str, Any]], line_no: int) -> Optional[Dict[str, Any]]:
    for n in nodes:
        if int(n.get("line_no", -1)) == int(line_no):
            return n
    return None


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}