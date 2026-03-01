# src/beacon_system/agents/verifier.py
# -*- coding: utf-8 -*-

"""
Verifier Agent (Constraints-only)

Design goals:
- Consume ONLY: code(str) + Constraints
- Perform ONLY: structural checks (required_calls/symbols + match_specs + forbidden_specs)
- Produce: VerifierReport with violations + revision directives
- NO Beacon reasoning, NO callgraph inference, NO env access

Notes:
- MatchSpec MUST be stable_json serializable by contract.
- This module delegates AST pattern matching to logic.matchers.match to keep one matcher implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import ast
import hashlib
import json

from ..types import Constraints, VerifierReport, Violation, Directive
from ..logic import matchers


# ----------------------------
# Helpers (deterministic)
# ----------------------------

def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _hash_stable(obj: Any) -> str:
    h = hashlib.sha256(_stable_json(obj).encode("utf-8")).hexdigest()
    return h


def _sorted_unique(xs: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(set([x for x in xs if x is not None and str(x).strip() != ""])))


def _safe_parse(code: str) -> Tuple[Optional[ast.AST], Optional[str]]:
    try:
        return ast.parse(code), None
    except SyntaxError as e:
        return None, f"SyntaxError: {e.msg} at line {e.lineno}, col {e.offset}"
    except Exception as e:
        return None, f"ParseError: {repr(e)}"


def _spec_to_ref(spec: Any) -> Dict[str, Any]:
    if isinstance(spec, dict):
        return spec
    d = getattr(spec, "__dict__", None)
    if isinstance(d, dict):
        return d
    return {"repr": repr(spec)}


# ----------------------------
# Core checks
# ----------------------------

def _check_required_calls(code_ast: ast.AST, required_calls: Sequence[str]) -> Tuple[int, int, List[Violation], List[Directive]]:
    """
    required_calls: list[str] of call names that must appear in code (approximate structural presence).
    Matching uses MatchSpec HasCall for consistency.
    """
    total = len(required_calls)
    hit = 0
    violations: List[Violation] = []
    directives: List[Directive] = []

    for call_name in required_calls:
        spec = {"type": "HasCall", "call_name": call_name}
        found = matchers.match(code_ast, spec)
        if found:
            hit += 1
        else:
            violations.append(
                Violation(
                    kind="missing_call",
                    detail=f"Required call not found: {call_name}",
                    spec_ref={"required_call": call_name},
                )
            )
            directives.append(
                Directive(
                    action="insert_call",
                    payload={"call_name": call_name, "position_hint": "inside_target_impl"},
                )
            )
    return hit, total, violations, directives


def _check_required_symbols(code_ast: ast.AST, required_symbols: Sequence[str]) -> Tuple[int, int, List[Violation], List[Directive]]:
    """
    required_symbols: names that should appear grounded in code.
    Matching uses MatchSpec HasName for consistency.
    """
    total = len(required_symbols)
    hit = 0
    violations: List[Violation] = []
    directives: List[Directive] = []

    for name in required_symbols:
        spec = {"type": "HasName", "name": name}
        found = matchers.match(code_ast, spec)
        if found:
            hit += 1
        else:
            violations.append(
                Violation(
                    kind="ungrounded_symbol",
                    detail=f"Required symbol not found: {name}",
                    spec_ref={"required_symbol": name},
                )
            )
            directives.append(
                Directive(
                    action="ground_symbol",
                    payload={"name": name, "hint": "import_or_reference_existing_symbol"},
                )
            )
    return hit, total, violations, directives


def _check_forbidden_specs(code_ast: ast.AST, forbidden_specs: Sequence[Any]) -> Tuple[List[Violation], List[Directive]]:
    """
    forbidden_specs: MatchSpec list. Any match -> violation.
    """
    violations: List[Violation] = []
    directives: List[Directive] = []

    for sp in forbidden_specs:
        spec_ref = _spec_to_ref(sp)
        found = matchers.match(code_ast, sp)
        if found:
            violations.append(
                Violation(
                    kind="forbidden_pattern",
                    detail=f"Forbidden spec matched: {spec_ref.get('type', 'MatchSpec')}",
                    spec_ref={"spec": spec_ref, "matches": [getattr(v, "__dict__", {"repr": repr(v)}) for v in found[:5]]},
                )
            )
            directives.append(
                Directive(
                    action="remove_pattern",
                    payload={"spec": spec_ref, "position_hint": "inside_target_impl"},
                )
            )
    return violations, directives


def _check_match_specs(code_ast: ast.AST, match_specs: Sequence[Any]) -> Tuple[List[Violation], List[Directive]]:
    """
    match_specs: optional specs that should be satisfied.
    Semantics:
      - If spec type implies presence (e.g., HasCall/HasImport/HasName/AssignFromCall/CallChain),
        then missing => violation.
      - If spec type implies forbiddance (e.g., ForbidCall/ForbidPattern),
        then match => violation.
    """
    violations: List[Violation] = []
    directives: List[Directive] = []

    for sp in match_specs:
        spec_ref = _spec_to_ref(sp)
        spec_type = spec_ref.get("type") if isinstance(spec_ref, dict) else None

        found = matchers.match(code_ast, sp)

        # Presence-required types
        if spec_type in {"HasCall", "HasImport", "HasName", "AssignFromCall", "CallChain"}:
            if not found:
                violations.append(
                    Violation(
                        kind="missing_match_spec",
                        detail=f"MatchSpec required but not satisfied: {spec_type}",
                        spec_ref={"spec": spec_ref},
                    )
                )
                directives.append(
                    Directive(
                        action="satisfy_spec",
                        payload={"spec": spec_ref, "position_hint": "inside_target_impl"},
                    )
                )

        # Forbidden types
        elif spec_type in {"ForbidCall", "ForbidPattern"}:
            if found:
                violations.append(
                    Violation(
                        kind="forbidden_pattern",
                        detail=f"Forbidden MatchSpec triggered: {spec_type}",
                        spec_ref={"spec": spec_ref, "matches": [getattr(v, "__dict__", {"repr": repr(v)}) for v in found[:5]]},
                    )
                )
                directives.append(
                    Directive(
                        action="remove_pattern",
                        payload={"spec": spec_ref, "position_hint": "inside_target_impl"},
                    )
                )
        else:
            # Unknown spec types: conservative behavior
            # - If it matches, we record info only (no violation) unless explicitly forbidden elsewhere.
            # - If it doesn't match, we do not enforce.
            continue

    return violations, directives


# ----------------------------
# Public API (contract)
# ----------------------------

def check(code: str, constraints: Constraints) -> VerifierReport:
    """
    Constraints-only verification.

    - Parse code into AST.
    - required_calls: presence check via matchers.HasCall
    - required_symbols: presence check via matchers.HasName
    - forbidden_specs: any match => violation
    - match_specs: presence-required vs forbidden types enforced
    """
    code_ast, parse_err = _safe_parse(code)

    # schema-tolerant: logic Constraints may not have `version`
    def g(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    meta = g(constraints, "meta", {}) or {}
    version = g(constraints, "version", None)
    if version is None:
        version = (
            meta.get("constraints_version")
            or meta.get("schema_version")
            or meta.get("version")
            or "mvp-0.1"
        )

    constraints_hash = _hash_stable(
        {
            "version": version,
            "required_symbols": list(g(constraints, "required_symbols", ()) or ()),
            "required_calls": list(g(constraints, "required_calls", ()) or ()),
            "forbidden_specs": [_spec_to_ref(s) for s in (g(constraints, "forbidden_specs", ()) or ())],
            "match_specs": [_spec_to_ref(s) for s in (g(constraints, "match_specs", ()) or ())],
            "meta": meta,
        }
    )

    if code_ast is None:
        v = Violation(kind="syntax_error", detail=parse_err or "syntax error", spec_ref={"phase": "parse"})
        d = Directive(action="fix_syntax", payload={"hint": "return_valid_python_code_only"})
        return VerifierReport(
            ok=False,
            coverage={
                "required_calls_hit": 0,
                "required_calls_total": len(constraints.required_calls),
                "required_symbols_hit": 0,
                "required_symbols_total": len(constraints.required_symbols),
            },
            violations=(v,),
            directives=(d,),
            meta={
                "constraints_hash": constraints_hash,
                "verifier_version": "0.1.0",
            },
        )

    required_calls = _sorted_unique(constraints.required_calls)
    required_symbols = _sorted_unique(constraints.required_symbols)

    rc_hit, rc_total, rc_viol, rc_dir = _check_required_calls(code_ast, required_calls)
    rs_hit, rs_total, rs_viol, rs_dir = _check_required_symbols(code_ast, required_symbols)

    forb_viol, forb_dir = _check_forbidden_specs(code_ast, constraints.forbidden_specs)
    ms_viol, ms_dir = _check_match_specs(code_ast, constraints.match_specs)

    violations: List[Violation] = []
    directives: List[Directive] = []
    violations.extend(rc_viol)
    violations.extend(rs_viol)
    violations.extend(forb_viol)
    violations.extend(ms_viol)
    directives.extend(rc_dir)
    directives.extend(rs_dir)
    directives.extend(forb_dir)
    directives.extend(ms_dir)

    ok = len(violations) == 0

    # Deterministic ordering of violations/directives to support stable outputs
    def _v_key(v: Violation) -> str:
        return _stable_json({"kind": v.kind, "detail": v.detail, "spec_ref": v.spec_ref})

    def _d_key(d: Directive) -> str:
        return _stable_json({"action": d.action, "payload": d.payload})

    violations_sorted = tuple(sorted(violations, key=_v_key))
    directives_sorted = tuple(sorted(directives, key=_d_key))

    return VerifierReport(
        ok=ok,
        coverage={
            "required_calls_hit": rc_hit,
            "required_calls_total": rc_total,
            "required_symbols_hit": rs_hit,
            "required_symbols_total": rs_total,
        },
        violations=violations_sorted,
        directives=directives_sorted,
        meta={
            "constraints_hash": constraints_hash,
            "verifier_version": "0.1.0",
        },
    )