# src/beacon_system/logic/rules_local.py
# -*- coding: utf-8 -*-
"""
rules_local.py

Local Beacon Logic (MVP):
- L-OUT: collect output-root nodes (Return / Yield / Print)
- L-DEP: backward dependency closure from output expressions (best-effort, intra-function)
- L-VAL: validation/guard filtering (early-exit branches -> forbidden)
- L-RED: local reduction (drop trivial nodes; deterministic via stable criteria)

Interface:
    apply_local(state: ReasoningState, func_key: FuncKey) -> None

IMPORTANT:
- This module is allowed to mutate ReasoningState only.
- It must NOT output IR or Constraints directly.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Optional, Set, Tuple

from .anchors import anchor_of, ast_kind, make_node_id, NodeID
from .state import (
    BeaconNode,
    BeaconEdge,
    EdgeKind,
    FuncKey,
    ProvenanceStep,
    ReasoningState,
    RuleName,
)


# ----------------------------
# AST helpers (MVP)
# ----------------------------

def _is_print_call(expr: ast.AST) -> bool:
    return isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id == "print"


def _collect_names_and_attrs(expr: ast.AST) -> Tuple[Set[str], Set[str]]:
    """
    Collect Name ids and Attribute attrs used in an expression subtree.
    """
    names: Set[str] = set()
    attrs: Set[str] = set()
    for n in ast.walk(expr):
        if isinstance(n, ast.Name):
            names.add(n.id)
        elif isinstance(n, ast.Attribute):
            attrs.add(n.attr)
    return names, attrs


def _stmt_defines_name(stmt: ast.stmt) -> Set[str]:
    """
    Return set of variable names defined by a statement (MVP).
    Covers: Assign, AnnAssign, AugAssign, For targets, With as targets.
    """
    defs: Set[str] = set()

    def add_target(t: ast.AST) -> None:
        if isinstance(t, ast.Name):
            defs.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for elt in t.elts:
                add_target(elt)

    if isinstance(stmt, ast.Assign):
        for t in stmt.targets:
            add_target(t)
    elif isinstance(stmt, ast.AnnAssign):
        add_target(stmt.target)
    elif isinstance(stmt, ast.AugAssign):
        add_target(stmt.target)
    elif isinstance(stmt, ast.For):
        add_target(stmt.target)
    elif isinstance(stmt, ast.AsyncFor):
        add_target(stmt.target)
    elif isinstance(stmt, ast.With):
        for item in stmt.items:
            if item.optional_vars is not None:
                add_target(item.optional_vars)
    elif isinstance(stmt, ast.AsyncWith):
        for item in stmt.items:
            if item.optional_vars is not None:
                add_target(item.optional_vars)

    return defs


def _flatten_function_body(fn_node: ast.AST) -> List[ast.stmt]:
    """
    Flatten only the top-level body statements of a function (MVP).
    We do not descend into nested defs/classes; we also keep If/For as single statements here.
    """
    body = getattr(fn_node, "body", None)
    if not isinstance(body, list):
        return []
    out: List[ast.stmt] = []
    for s in body:
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(s, ast.stmt):
            out.append(s)
    return out


def _make_stmt_node(state: ReasoningState, stmt: ast.AST, file: str, qualname: str) -> BeaconNode:
    anch = anchor_of(stmt, file=file, qualname=qualname)
    kind = ast_kind(stmt)
    idx = state.next_local_index(anch, kind)
    nid = make_node_id(anch, kind, idx)
    return BeaconNode(node_id=nid, anchor=anch, kind=kind, code=None, meta={})


def _make_expr_node(state: ReasoningState, expr: ast.AST, file: str, qualname: str, *, meta: Optional[dict] = None) -> BeaconNode:
    anch = anchor_of(expr, file=file, qualname=qualname)
    kind = ast_kind(expr)
    idx = state.next_local_index(anch, kind)
    nid = make_node_id(anch, kind, idx)
    return BeaconNode(node_id=nid, anchor=anch, kind=kind, code=None, meta=meta or {})


def _parse_funckey(key: FuncKey) -> Tuple[str, str]:
    s = str(key)
    if "::" not in s:
        return ("", s)
    file, qual = s.split("::", 1)
    return file, qual


import ast
from typing import Optional, Tuple

def _extract_assign_from_call(stmt: ast.stmt) -> Optional[Tuple[str, str]]:
    """
    If stmt matches:
      - x = foo(...)
      - x: T = foo(...)
    return (var, call_name). Otherwise None.
    """
    def call_name_of(call: ast.Call) -> Optional[str]:
        if isinstance(call.func, ast.Name):
            return call.func.id
        if isinstance(call.func, ast.Attribute):
            return call.func.attr
        return None

    # x = foo(...)
    if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
        if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            cn = call_name_of(stmt.value)
            if cn:
                return (stmt.targets[0].id, cn)

    # x: T = foo(...)
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call):
        if isinstance(stmt.target, ast.Name):
            cn = call_name_of(stmt.value)
            if cn:
                return (stmt.target.id, cn)

    return None


def _set_node_meta_assign_from_call(
    state: "ReasoningState",
    node_id: "NodeID",
    var: str,
    call_name: str,
) -> None:
    """
    BeaconNode is frozen; overwrite node with updated meta.
    """
    node = state.get_node(node_id)
    if node is None:
        return
    new_meta = dict(node.meta)
    new_meta["assign_from_call"] = {"var": var, "call_name": call_name}
    # keep any existing meta keys
    new_node = BeaconNode(
        node_id=node.node_id,
        anchor=node.anchor,
        kind=node.kind,
        code=node.code,
        meta=new_meta,
    )
    state.add_node(
        new_node,
        prov=ProvenanceStep(rule=RuleName.L_DEP, note="assign_from_call_meta"),
        overwrite=True,
    )

# ----------------------------
# L-OUT
# ----------------------------

def _apply_l_out(state: ReasoningState, fn_node: ast.AST, file: str, qualname: str) -> List[NodeID]:
    """
    Add output-root nodes: Return / Yield / print(...) statement expression.
    Returns list of node_ids added/seen as output roots.
    """
    out_ids: List[NodeID] = []

    # Walk within function body
    for n in ast.walk(fn_node):
        # Ignore nested functions/classes
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not fn_node:
            continue

        if isinstance(n, ast.Return):
            bn = _make_stmt_node(state, n, file, qualname)
            state.add_node(bn, prov=ProvenanceStep(rule=RuleName.L_OUT, note="return"))
            out_ids.append(bn.node_id)

        elif isinstance(n, ast.Yield) or isinstance(n, ast.YieldFrom):
            bn = _make_expr_node(state, n, file, qualname, meta={"output": "yield"})
            state.add_node(bn, prov=ProvenanceStep(rule=RuleName.L_OUT, note="yield"))
            out_ids.append(bn.node_id)

        elif isinstance(n, ast.Expr) and _is_print_call(n.value):
            bn = _make_stmt_node(state, n, file, qualname)
            bn2 = BeaconNode(
                node_id=bn.node_id,
                anchor=bn.anchor,
                kind=bn.kind,
                code=bn.code,
                meta={"output": "print"},
            )
            state.add_node(bn2, prov=ProvenanceStep(rule=RuleName.L_OUT, note="print"), overwrite=True)
            out_ids.append(bn2.node_id)

    return out_ids


# ----------------------------
# L-DEP (best-effort backward closure)
# ----------------------------

def _apply_l_dep(
    state: ReasoningState,
    fn_node: ast.AST,
    file: str,
    qualname: str,
    output_node_ids: List[NodeID],
) -> None:
    """
    Backward dependency closure:
    - collect names used in output expressions
    - find nearest definitions in top-level body (reverse scan)
    - add defining statements as beacon nodes; add DATA edges from def->output
    - iteratively expand: if defining stmt uses other names, keep expanding
    """
    body = _flatten_function_body(fn_node)
    if not body:
        return

    # Build a quick map from stmt index -> defined names
    defines: List[Set[str]] = [_stmt_defines_name(s) for s in body]

    # Helper to get expression corresponding to an output node (best-effort)
    # We re-scan AST because NodeID->AST mapping isn't in MVP yet.
    # Later: keep a NodeID->AST pointer map in ReasoningState.
    output_exprs: List[ast.AST] = []
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Return) and n.value is not None:
            output_exprs.append(n.value)
        elif isinstance(n, (ast.Yield, ast.YieldFrom)) and getattr(n, "value", None) is not None:
            output_exprs.append(n.value)
        elif isinstance(n, ast.Expr) and _is_print_call(n.value):
            # print(args...) -> treat args as outputs
            for arg in n.value.args:
                output_exprs.append(arg)

    seed_names: Set[str] = set()
    seed_attrs: Set[str] = set()
    for expr in output_exprs:
        names, attrs = _collect_names_and_attrs(expr)
        seed_names |= names
        seed_attrs |= attrs

    # Record attributes as symbols (MVP)
    state.symbols.attrs |= seed_attrs

    # Iterative expansion queue
    needed: Set[str] = set(seed_names)
    resolved: Set[str] = set()

    # For determinism, we loop until no progress; resolution uses reverse scan each time
    while True:
        progress = False
        unresolved = sorted([n for n in needed if n not in resolved])
        if not unresolved:
            break

        for name in unresolved:
            # Find nearest definition by scanning body backwards
            def_stmt = None
            def_idx = None
            for i in range(len(body) - 1, -1, -1):
                if name in defines[i]:
                    def_stmt = body[i]
                    def_idx = i
                    break
            if def_stmt is None:
                # treat as global / free var
                state.symbols.globals.add(name)
                resolved.add(name)
                continue

            # Add beacon node for definition statement
            def_node = _make_stmt_node(state, def_stmt, file, qualname)
            state.add_node(def_node, prov=ProvenanceStep(rule=RuleName.L_DEP, note=f"define:{name}"))

            afc = _extract_assign_from_call(def_stmt)
            if afc is not None:
                var, call_name = afc
                # Only tag if it matches the variable we are currently resolving (keeps it precise)
                if var == name:
                    _set_node_meta_assign_from_call(state, def_node.node_id, var=var, call_name=call_name)
                    # Ensure symbol table calls contains it (deterministic, idempotent)
                    state.symbols.calls.add(call_name)

            # Add DATA edges from def_node -> each output root (MVP)
            for out_id in output_node_ids:
                edge = BeaconEdge(src=def_node.node_id, dst=out_id, kind=EdgeKind.DATA, meta={"var": name})
                state.add_edge(edge, prov=ProvenanceStep(rule=RuleName.L_DEP, src=def_node.node_id, note="dep_to_output"))

            # Expand dependencies from the RHS / stmt subtree
            # We approximate by collecting Name/Attribute in the defining statement.
            rhs_names: Set[str] = set()
            rhs_attrs: Set[str] = set()
            for sub in ast.walk(def_stmt):
                if isinstance(sub, ast.Name):
                    rhs_names.add(sub.id)
                elif isinstance(sub, ast.Attribute):
                    rhs_attrs.add(sub.attr)
                elif isinstance(sub, ast.Call):
                    # record calls for potential global reasoning
                    if isinstance(sub.func, ast.Name):
                        state.symbols.calls.add(sub.func.id)
                    elif isinstance(sub.func, ast.Attribute):
                        state.symbols.calls.add(sub.func.attr)

            # Remove the defined name itself to avoid self-loop
            rhs_names.discard(name)

            # Update symbol table and needed set
            state.symbols.attrs |= rhs_attrs
            before = len(needed)
            needed |= rhs_names
            if len(needed) != before:
                progress = True

            resolved.add(name)

        if not progress:
            break


# ----------------------------
# L-VAL (validation / guard filtering)
# ----------------------------

def _is_early_exit_if(stmt: ast.If) -> bool:
    """
    MVP early-exit guard detection:
    - if <cond>: return ...
    - if <cond>: raise ...
    - if not <x>: return/raise
    - if x is None: return/raise
    We treat any If whose body contains Return/Raise and has no else (or empty else) as guard.
    """
    has_exit = any(isinstance(s, (ast.Return, ast.Raise)) for s in stmt.body)
    if not has_exit:
        return False
    if stmt.orelse:
        # If there's a meaningful else, we avoid classifying as simple guard in MVP
        # (later can refine)
        return False
    # Accept simple patterns
    return True


def _apply_l_val(state: ReasoningState, fn_node: ast.AST, file: str, qualname: str) -> None:
    """
    Mark nodes inside early-exit guards as forbidden.

    MVP:
    - Find If nodes meeting _is_early_exit_if
    - For each stmt in its body, create/find beacon node and mark forbidden
    - Optionally keep them in state.nodes but record forbidden set (preferred)
    """
    for n in ast.walk(fn_node):
        if isinstance(n, ast.If) and _is_early_exit_if(n):
            # Mark the If node itself as forbidden context
            if_node = _make_stmt_node(state, n, file, qualname)
            state.add_node(if_node, prov=ProvenanceStep(rule=RuleName.L_VAL, note="guard_if"))
            state.mark_forbidden(if_node.node_id, prov=ProvenanceStep(rule=RuleName.L_VAL, note="forbidden_if"))

            for s in n.body:
                if not isinstance(s, ast.stmt):
                    continue
                bn = _make_stmt_node(state, s, file, qualname)
                state.add_node(bn, prov=ProvenanceStep(rule=RuleName.L_VAL, src=if_node.node_id, note="guard_body"))
                state.mark_forbidden(bn.node_id, prov=ProvenanceStep(rule=RuleName.L_VAL, src=if_node.node_id, note="forbidden_guard_body"))


# ----------------------------
# L-RED (reduction)
# ----------------------------

def _is_trivial_stmt(stmt: ast.stmt) -> bool:
    """
    MVP triviality:
    - Pass
    - Expr(Constant) docstring-like or standalone literal
    """
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
        return True
    return False


def _apply_l_red(state: ReasoningState, file: str, qualname: str) -> None:
    """
    Deterministic local reduction:
    - Drop trivial nodes of the target function scope (Pass, Expr(Constant))
    - Keep forbidden nodes (still meaningful for verifier), but you may choose to drop them later in normalize/reduce.
    """
    to_drop: List[NodeID] = []
    for nid, node in state.nodes.items():
        if node.anchor.file != file or node.anchor.qualname != qualname:
            continue
        # Only attempt to classify statement kinds we know
        if node.kind == "Pass":
            to_drop.append(nid)
        elif node.kind == "Expr":
            # best effort: if meta indicates output=print keep; else drop as trivial constant expr
            if node.meta.get("output") == "print":
                continue
            # We cannot inspect original AST here; conservative drop for Expr w/o meta
            to_drop.append(nid)

    # Deterministic deletion order
    for nid in sorted(to_drop, key=str):
        # Do not drop if it's an output root or forbidden marker? MVP: keep forbidden, drop trivial regardless.
        # If you prefer: if nid in state.forbidden: continue
        state.nodes.pop(nid, None)
        # Remove from forbidden set if dropped
        state.forbidden.discard(nid)

    # Also prune edges that became hanging (do not do full normalize here; normalize.py will prune)
    state.edges = {e for e in state.edges if (e.src in state.nodes and e.dst in state.nodes)}


# ----------------------------
# Public API
# ----------------------------

def apply_local(state: ReasoningState, func_key: FuncKey) -> None:
    """
    Apply local reasoning rules in the fixed order:
        L-OUT -> L-DEP -> L-VAL (optional) -> L-RED

    This function is idempotent-ish for MVP: repeated calls will add nodes with new local_index
    if you rebuild anchors each time. In practice engine should call apply_local once per function.
    """
    fn_node = state.ast_index.get_function(func_key)
    if fn_node is None:
        return

    file, qualname = _parse_funckey(func_key)

    # L-OUT
    output_ids = _apply_l_out(state, fn_node, file, qualname)

    # L-DEP
    _apply_l_dep(state, fn_node, file, qualname, output_ids)

    # L-VAL
    if state.config.validation_filter:
        _apply_l_val(state, fn_node, file, qualname)

    # L-RED
    _apply_l_red(state, file, qualname)