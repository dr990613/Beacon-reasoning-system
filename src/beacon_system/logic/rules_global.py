# src/beacon_system/logic/rules_global.py
# -*- coding: utf-8 -*-
"""
rules_global.py

Global Beacon Logic (MVP):

- G-BASE: mark entry function's local beacons as "global_base"
- G-CALL: inline semantically relevant callees' local beacons into entry closure
          (heuristic relevance: assigned-from-call variable participates in output dependency,
           or call appears directly in return expression; fallback: all project-resolvable calls)

- G-RET / G-GLOB: stubs for MVP
- P-ENTRY: finalize entry-level closure (MVP: tagging only)

Single Source of Truth: This module lives in logic/, so it may call other logic rules.
Verifier/Generator MUST NOT re-implement any of this.

Interface:
    apply_global(state: ReasoningState, entry_key: FuncKey) -> None
"""

from __future__ import annotations

import ast
from dataclasses import replace
from typing import Dict, Iterable, List, Optional, Set, Tuple

from .anchors import anchor_of, ast_kind, make_node_id, NodeID
from .state import (
    BeaconEdge,
    BeaconNode,
    EdgeKind,
    FuncKey,
    ProvenanceStep,
    ReasoningState,
    RuleName,
)


# ----------------------------
# Small AST helpers
# ----------------------------

def _parse_funckey(key: FuncKey) -> Tuple[str, str]:
    """
    FuncKey format in state.ASTIndex.register_function():  "<file>::<qualname>"
    """
    s = str(key)
    if "::" not in s:
        return ("", s)
    file, qual = s.split("::", 1)
    return file, qual


def _get_call_name(call: ast.Call) -> Optional[str]:
    """
    Best-effort extraction of a call target name.
    - foo(...) -> "foo"
    - obj.foo(...) -> "foo"
    - pkg.mod.foo(...) -> "foo"
    """
    fn = call.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


def _collect_names(expr: ast.AST) -> Set[str]:
    """
    Collect Name identifiers used in an expression subtree.
    """
    out: Set[str] = set()
    for n in ast.walk(expr):
        if isinstance(n, ast.Name):
            out.add(n.id)
    return out


def _collect_return_calls(fn_node: ast.AST) -> Set[str]:
    """
    Collect call names that appear inside return expressions of a function.
    """
    calls: Set[str] = set()
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Return) and n.value is not None:
            for sub in ast.walk(n.value):
                if isinstance(sub, ast.Call):
                    name = _get_call_name(sub)
                    if name:
                        calls.add(name)
    return calls


def _assigned_from_calls(fn_node: ast.AST) -> Dict[str, str]:
    """
    Map var_name -> call_name for simple patterns:
        x = foo(...)
    """
    m: Dict[str, str] = {}
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            call_name = _get_call_name(n.value)
            if not call_name:
                continue
            # Only handle simple 'x = call(...)'
            if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                m[n.targets[0].id] = call_name
        elif isinstance(n, ast.AnnAssign) and isinstance(n.value, ast.Call) and isinstance(n.target, ast.Name):
            call_name = _get_call_name(n.value)
            if call_name:
                m[n.target.id] = call_name
    return m


def _used_in_outputs(fn_node: ast.AST) -> Set[str]:
    """
    Names that appear in return expressions (MVP output dependency seed).
    """
    used: Set[str] = set()
    for n in ast.walk(fn_node):
        if isinstance(n, ast.Return) and n.value is not None:
            used |= _collect_names(n.value)
    return used


def _resolve_project_funcs(state: ReasoningState, call_name: str) -> List[FuncKey]:
    """
    Resolve a call name to candidate FuncKeys in the indexed project.

    MVP heuristic:
    - exact qualname match (qual == call_name)
    - suffix match (qual endswith ".<call_name>")
    - method match (qual endswith ":<call_name>" if you use ":" in qualnames)
    """
    cands: List[FuncKey] = []
    for fk in state.ast_index.functions.keys():
        _, qual = _parse_funckey(fk)
        if qual == call_name or qual.endswith(f".{call_name}") or qual.endswith(f":{call_name}"):
            cands.append(fk)
    return cands


def _make_callsite_node(state: ReasoningState, call: ast.Call, file: str, qualname: str) -> BeaconNode:
    """
    Create a beacon node representing the callsite (MVP).
    """
    anch = anchor_of(call, file=file, qualname=qualname)
    kind = ast_kind(call)
    idx = state.next_local_index(anch, kind)
    nid = make_node_id(anch, kind, idx)
    meta = {"call_name": _get_call_name(call)}
    return BeaconNode(node_id=nid, anchor=anch, kind=kind, code=None, meta=meta)


def _tag_node_meta(state: ReasoningState, node_id: NodeID, **kwargs) -> None:
    """
    BeaconNode is frozen; tagging requires overwrite with updated meta.
    """
    node = state.get_node(node_id)
    if node is None:
        return
    new_meta = dict(node.meta)
    new_meta.update(kwargs)
    new_node = BeaconNode(
        node_id=node.node_id,
        anchor=node.anchor,
        kind=node.kind,
        code=node.code,
        meta=new_meta,
    )
    # Overwrite the stored node (keep provenance history)
    state.add_node(new_node, prov=ProvenanceStep(rule=RuleName.G_BASE, note="meta_tag"), overwrite=True)


# ----------------------------
# Global rules (MVP)
# ----------------------------

def apply_global(state: ReasoningState, entry_key: FuncKey) -> None:
    """
    Run MVP global reasoning on the entry function.

    This function assumes local reasoning has already been run for the entry function.
    If callees need local beacons, this function may call local rules (inside logic boundary).
    """
    if not state.config.enable_global:
        return

    entry_fn = state.ast_index.get_function(entry_key)
    if entry_fn is None:
        return

    entry_file, entry_qual = _parse_funckey(entry_key)

    # -------- G-BASE: tag entry function beacons as global base
    for nid, node in list(state.nodes.items()):
        if node.anchor.file == entry_file and node.anchor.qualname == entry_qual:
            # tag meta; preserve node_id
            new_meta = dict(node.meta)
            new_meta["scope"] = "global_base"
            new_node = BeaconNode(
                node_id=node.node_id,
                anchor=node.anchor,
                kind=node.kind,
                code=node.code,
                meta=new_meta,
            )
            state.add_node(
                new_node,
                prov=ProvenanceStep(rule=RuleName.G_BASE, note="entry_base"),
                overwrite=True,
            )

    # -------- G-CALL: find semantically relevant calls
    used_names = _used_in_outputs(entry_fn)
    var_to_call = _assigned_from_calls(entry_fn)
    return_calls = _collect_return_calls(entry_fn)

    relevant_call_names: Set[str] = set()

    # If a variable used in outputs comes from a call, include that call
    for var, call_name in var_to_call.items():
        if var in used_names:
            relevant_call_names.add(call_name)

    # Calls directly in return expressions are relevant
    relevant_call_names |= return_calls

    # Fallback: if nothing found, include all calls we can see (conservative)
    all_seen_calls: Set[str] = set()
    for n in ast.walk(entry_fn):
        if isinstance(n, ast.Call):
            cn = _get_call_name(n)
            if cn:
                all_seen_calls.add(cn)
    if not relevant_call_names:
        relevant_call_names = set(all_seen_calls)

    # Optional filter: if state.symbols.calls exists, intersect to reduce noise
    if state.symbols.calls:
        # Keep calls that either appear in symbols.calls or can be resolved to project funcs
        filtered: Set[str] = set()
        for cn in relevant_call_names:
            if cn in state.symbols.calls or _resolve_project_funcs(state, cn):
                filtered.add(cn)
        if filtered:
            relevant_call_names = filtered

    # -------- Inline callees
    # We will:
    # 1) create callsite nodes for each relevant call occurrence (MVP)
    # 2) resolve callee FuncKey candidates from project index
    # 3) ensure local beacons for callee exist (call rules_local.apply_local if available)
    # 4) tag callee nodes as inlined, and add CALL edges from callsite->callee representative node

    # Import locally to avoid circulars; rules_local is within logic so it is allowed.
    try:
        from .rules_local import apply_local  # type: ignore
    except Exception:
        apply_local = None  # type: ignore

    inlined_count = 0

    for n in ast.walk(entry_fn):
        if not isinstance(n, ast.Call):
            continue

        call_name = _get_call_name(n)
        if not call_name or call_name not in relevant_call_names:
            continue

        # Create a callsite beacon node (even if local rules didn't include it)
        call_node = _make_callsite_node(state, n, entry_file, entry_qual)
        state.add_node(
            call_node,
            prov=ProvenanceStep(rule=RuleName.G_CALL, note="callsite"),
            overwrite=False,
        )

        # Resolve callee candidates in project
        callee_keys = _resolve_project_funcs(state, call_name)
        if not callee_keys:
            continue

        # Deterministically pick the first candidate (stable by lexical order)
        callee_keys = sorted(callee_keys, key=str)
        callee_key = callee_keys[0]
        callee_file, callee_qual = _parse_funckey(callee_key)

        # Ensure callee local beacons exist (if local rule is available)
        if apply_local is not None:
            callee_fn = state.ast_index.get_function(callee_key)
            if callee_fn is not None:
                apply_local(state, callee_key)  # Local reasoning is still within logic/

        # Collect callee beacon nodes currently in state
        callee_node_ids: List[NodeID] = [
            nid for nid, bn in state.nodes.items()
            if bn.anchor.file == callee_file and bn.anchor.qualname == callee_qual
        ]
        callee_node_ids = sorted(callee_node_ids, key=str)
        if not callee_node_ids:
            continue

        # Tag callee nodes as inlined into entry closure (MVP)
        for nid in callee_node_ids:
            node = state.get_node(nid)
            if node is None:
                continue
            new_meta = dict(node.meta)
            new_meta["inlined_into"] = str(entry_key)
            new_meta["inlined_from"] = str(callee_key)
            new_node = BeaconNode(
                node_id=node.node_id,
                anchor=node.anchor,
                kind=node.kind,
                code=node.code,
                meta=new_meta,
            )
            state.add_node(
                new_node,
                prov=ProvenanceStep(rule=RuleName.G_CALL, src=call_node.node_id, note="inline"),
                overwrite=True,
            )

        # Add one representative CALL edge: callsite -> callee_first_beacon
        rep = callee_node_ids[0]
        edge = BeaconEdge(
            src=call_node.node_id,
            dst=rep,
            kind=EdgeKind.CALL,
            meta={"call_name": call_name, "callee": str(callee_key)},
        )
        state.add_edge(edge, prov=ProvenanceStep(rule=RuleName.G_CALL, src=call_node.node_id, note="call_edge"))

        inlined_count += 1
        if state.config.max_global_inline is not None and inlined_count >= state.config.max_global_inline:
            break

    # -------- G-RET (stub)
    # TODO: Add return-flow edges (EdgeKind.RET) once return reasoning is implemented.

    # -------- G-GLOB (stub)
    # TODO: Add conservative global-state interactions (EdgeKind.GLOBAL) once implemented.

    # -------- P-ENTRY: finalize entry closure (MVP tagging)
    # Tag all nodes that are either entry base or inlined_into entry as part of the entry closure.
    for nid, node in list(state.nodes.items()):
        if node.meta.get("scope") == "global_base" or node.meta.get("inlined_into") == str(entry_key):
            new_meta = dict(node.meta)
            new_meta["entry_closure"] = str(entry_key)
            new_node = BeaconNode(
                node_id=node.node_id,
                anchor=node.anchor,
                kind=node.kind,
                code=node.code,
                meta=new_meta,
            )
            state.add_node(
                new_node,
                prov=ProvenanceStep(rule=RuleName.P_ENTRY, note="entry_finalize"),
                overwrite=True,
            )