# src/beacon_system/agents/checks.py
# -*- coding: utf-8 -*-

"""
Agent-side checks.

Scope:
1. logic output acceptance:
   validate BeaconIR + Constraints are minimally usable for downstream agent steps.

2. Beacon usage check:
   verify generated code actually reflects Beacon-required symbols/calls,
   instead of merely claiming Beacon was used.

Non-goals:
- no logic rebuilding
- no execution
- no AST-heavy semantic tracing
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

from ..types import (
    BeaconIR,
    BeaconUsageReport,
    Constraints,
    LogicAcceptanceReport,
)


def _safe_text(text: Optional[str]) -> str:
    return str(text or "")


def _contains_symbol(code: str, symbol: str) -> bool:
    if not symbol or not symbol.strip():
        return False
    pattern = r"\b" + re.escape(symbol.strip()) + r"\b"
    return re.search(pattern, code) is not None


def _contains_call(code: str, call_name: str) -> bool:
    if not call_name or not call_name.strip():
        return False
    pattern = re.escape(call_name.strip()) + r"\s*\("
    return re.search(pattern, code) is not None


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _get_entry_dict(ir: Any) -> dict:
    entry = _get_attr(ir, "entry", {})
    if isinstance(entry, dict):
        return entry
    return {}


def _get_nodes(ir: Any) -> Sequence[Any]:
    nodes = _get_attr(ir, "nodes", ())
    if nodes is None:
        return ()
    return nodes


def _get_edges(ir: Any) -> Sequence[Any]:
    edges = _get_attr(ir, "edges", ())
    if edges is None:
        return ()
    return edges


def _get_forbidden(ir: Any) -> Sequence[Any]:
    forbidden = _get_attr(ir, "forbidden", ())
    if forbidden is None:
        return ()
    return forbidden


def _get_constraint_items(constraints: Any, name: str) -> Sequence[Any]:
    items = _get_attr(constraints, name, ())
    if items is None:
        return ()
    return items


@dataclass
class LogicOutputChecker:
    """
    Minimal acceptance checker for logic outputs.

    Purpose:
    - ensure downstream planning/generation receives minimally valid IR+Constraints
    - fail early when logic output is obviously incomplete
    """
    print_io: bool = False

    def _print(self, message: str) -> None:
        if self.print_io:
            print(f"[LogicOutputChecker] {message}")

    def check(self, ir: BeaconIR, constraints: Constraints) -> LogicAcceptanceReport:
        issues: List[str] = []
        warnings: List[str] = []

        self._print("start logic acceptance check")

        # ---------- IR checks ----------
        if ir is None:
            issues.append("BeaconIR is missing.")
        else:
            ir_version = _get_attr(ir, "version", None)
            if ir_version is None:
                warnings.append("BeaconIR.version is missing.")
            elif not str(ir_version).strip():
                warnings.append("BeaconIR.version is empty.")

            entry = _get_entry_dict(ir)
            if not entry:
                warnings.append("BeaconIR.entry is missing or empty.")
            else:
                entry_file = str(entry.get("file") or "").strip()
                entry_qualname = str(entry.get("qualname") or "").strip()
                if not entry_file:
                    warnings.append("BeaconIR.entry.file is empty.")
                if not entry_qualname:
                    warnings.append("BeaconIR.entry.qualname is empty.")

            nodes = _get_nodes(ir)
            if len(nodes) == 0:
                warnings.append("BeaconIR.nodes is empty.")
            else:
                bad_nodes = 0
                for node in nodes:
                    node_id = _get_attr(node, "id", "")
                    node_kind = _get_attr(node, "kind", "")
                    anchor = _get_attr(node, "anchor", None)

                    if not str(node_id or "").strip():
                        bad_nodes += 1
                    if not str(node_kind or "").strip():
                        bad_nodes += 1
                    if anchor is None:
                        bad_nodes += 1

                if bad_nodes > 0:
                    warnings.append(f"BeaconIR has invalid node fields: count={bad_nodes}.")

            edges = _get_edges(ir)
            if len(edges) == 0:
                warnings.append("BeaconIR.edges is empty.")
            else:
                bad_edges = 0
                node_ids = {str(_get_attr(n, 'id', '')) for n in nodes}
                for edge in edges:
                    edge_kind = _get_attr(edge, "kind", "")
                    edge_src = str(_get_attr(edge, "src", ""))
                    edge_dst = str(_get_attr(edge, "dst", ""))

                    if not str(edge_kind or "").strip():
                        bad_edges += 1
                    if node_ids and edge_src not in node_ids:
                        bad_edges += 1
                    if node_ids and edge_dst not in node_ids:
                        bad_edges += 1

                if bad_edges > 0:
                    warnings.append(f"BeaconIR has edges with unresolved refs: count={bad_edges}.")

        # ---------- Constraints checks ----------
        if constraints is None:
            issues.append("Constraints is missing.")
        else:
            constraints_version = _get_attr(constraints, "version", None)
            if constraints_version is None:
                warnings.append("Constraints.version is missing.")
            elif not str(constraints_version).strip():
                warnings.append("Constraints.version is empty.")

            required_symbols = _get_constraint_items(constraints, "required_symbols")
            required_calls = _get_constraint_items(constraints, "required_calls")
            forbidden_specs = _get_constraint_items(constraints, "forbidden_specs")
            match_specs = _get_constraint_items(constraints, "match_specs")

            if (
                len(required_symbols) == 0
                and len(required_calls) == 0
                and len(forbidden_specs) == 0
                and len(match_specs) == 0
            ):
                warnings.append("Constraints are empty; downstream guidance may be too weak.")

        ok = len(issues) == 0

        self._print(
            f"logic acceptance done: ok={ok} issues={len(issues)} warnings={len(warnings)}"
        )

        return LogicAcceptanceReport(
            ok=ok,
            issues=tuple(issues),
            warnings=tuple(warnings),
            meta={
                "checker": "LogicOutputChecker",
                "node_count": len(_get_nodes(ir)) if ir is not None else 0,
                "edge_count": len(_get_edges(ir)) if ir is not None else 0,
                "required_symbols_count": len(_get_constraint_items(constraints, "required_symbols")) if constraints is not None else 0,
                "required_calls_count": len(_get_constraint_items(constraints, "required_calls")) if constraints is not None else 0,
                "forbidden_specs_count": len(_get_constraint_items(constraints, "forbidden_specs")) if constraints is not None else 0,
                "match_specs_count": len(_get_constraint_items(constraints, "match_specs")) if constraints is not None else 0,
            },
        )


@dataclass
class BeaconUsageChecker:
    """
    Minimal Beacon usage checker.

    Current policy:
    - treat required_symbols / required_calls as the primary hard evidence
    - optionally include note-level warnings from IR when constraints are weak
    """
    print_io: bool = False

    def _print(self, message: str) -> None:
        if self.print_io:
            print(f"[BeaconUsageChecker] {message}")

    def check(
        self,
        *,
        code: str,
        ir: BeaconIR,
        constraints: Constraints,
    ) -> BeaconUsageReport:
        code = _safe_text(code)

        self._print("start beacon usage check")

        required_symbols: Sequence[str] = tuple(_get_constraint_items(constraints, "required_symbols"))
        required_calls: Sequence[str] = tuple(_get_constraint_items(constraints, "required_calls"))

        used_required_symbols: List[str] = []
        missing_symbols: List[str] = []

        for sym in required_symbols:
            if _contains_symbol(code, sym):
                used_required_symbols.append(sym)
            else:
                missing_symbols.append(sym)

        used_required_calls: List[str] = []
        missing_calls: List[str] = []

        for call_name in required_calls:
            if _contains_call(code, call_name):
                used_required_calls.append(call_name)
            else:
                missing_calls.append(call_name)

        notes: List[str] = []

        if len(required_symbols) == 0 and len(required_calls) == 0:
            notes.append(
                "Constraints expose no required_symbols/required_calls; Beacon usage evidence is weak."
            )

        if ir is not None and len(_get_nodes(ir)) == 0:
            notes.append("BeaconIR has no nodes; usage check relies only on constraints.")

        if ir is not None and len(_get_forbidden(ir)) > 0:
            notes.append("BeaconIR contains forbidden nodes; ensure generation avoided reconstructing them.")

        ok = (len(missing_symbols) == 0 and len(missing_calls) == 0)

        self._print(
            f"beacon usage done: ok={ok} "
            f"used_symbols={len(used_required_symbols)}/{len(required_symbols)} "
            f"used_calls={len(used_required_calls)}/{len(required_calls)}"
        )

        return BeaconUsageReport(
            ok=ok,
            used_required_symbols=tuple(used_required_symbols),
            used_required_calls=tuple(used_required_calls),
            missing_symbols=tuple(missing_symbols),
            missing_calls=tuple(missing_calls),
            notes=tuple(notes),
            meta={
                "checker": "BeaconUsageChecker",
                "code_len": len(code),
                "required_symbols_total": len(required_symbols),
                "required_calls_total": len(required_calls),
                "ir_node_count": len(_get_nodes(ir)) if ir is not None else 0,
                "ir_edge_count": len(_get_edges(ir)) if ir is not None else 0,
            },
        )


def check_logic_outputs(
    ir: BeaconIR,
    constraints: Constraints,
    *,
    print_io: bool = False,
) -> LogicAcceptanceReport:
    checker = LogicOutputChecker(print_io=print_io)
    return checker.check(ir=ir, constraints=constraints)


def check_beacon_usage(
    *,
    code: str,
    ir: BeaconIR,
    constraints: Constraints,
    print_io: bool = False,
) -> BeaconUsageReport:
    checker = BeaconUsageChecker(print_io=print_io)
    return checker.check(code=code, ir=ir, constraints=constraints)