# src/beacon_system/agents/verifier.py
# -*- coding: utf-8 -*-

"""
Minimal constraints verifier.

Scope:
- ONLY verify generated code against Constraints.
- DO NOT reconstruct reasoning.
- DO NOT call logic.
- DO NOT execute code.

Current strategy:
- lightweight text/pattern based checks
- stable, explainable, easy to debug
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..types import Constraints, Directive, Violation, VerifierResult


def _safe_text(code: Optional[str]) -> str:
    return str(code or "")


def _contains_symbol(code: str, symbol: str) -> bool:
    """
    Loose symbol presence check.

    Examples:
    - Time
    - FixedOffset
    - timezone
    """
    if not symbol or not symbol.strip():
        return True

    pattern = r"\b" + re.escape(symbol.strip()) + r"\b"
    return re.search(pattern, code) is not None


def _contains_call(code: str, call_name: str) -> bool:
    """
    Loose function/method call presence check.

    Examples:
    - divmod
    - localize
    - zone.localize

    Strategy:
    - allow optional whitespace before '('
    - treat dotted name literally
    """
    if not call_name or not call_name.strip():
        return True

    pattern = re.escape(call_name.strip()) + r"\s*\("
    return re.search(pattern, code) is not None


def _match_text_spec(code: str, spec: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Supported text-like spec forms:

    1) {"type": "contains", "value": "..."}
    2) {"type": "not_contains", "value": "..."}
    3) {"type": "regex", "pattern": "..."}
    4) {"type": "not_regex", "pattern": "..."}
    5) {"contains": "..."} / {"regex": "..."} (compact legacy style)
    """
    spec_type = str(spec.get("type") or "").strip()

    if not spec_type:
        if "contains" in spec:
            spec_type = "contains"
            value = str(spec.get("contains") or "")
        elif "not_contains" in spec:
            spec_type = "not_contains"
            value = str(spec.get("not_contains") or "")
        elif "regex" in spec:
            spec_type = "regex"
            value = str(spec.get("regex") or "")
        elif "not_regex" in spec:
            spec_type = "not_regex"
            value = str(spec.get("not_regex") or "")
        else:
            value = str(spec.get("value") or spec.get("pattern") or "")
    else:
        value = str(spec.get("value") or spec.get("pattern") or "")

    if spec_type == "contains":
        ok = value in code
        return ok, f"expected code to contain text: {value!r}"

    if spec_type == "not_contains":
        ok = value not in code
        return ok, f"expected code to avoid text: {value!r}"

    if spec_type == "regex":
        ok = re.search(value, code) is not None
        return ok, f"expected code to match regex: {value!r}"

    if spec_type == "not_regex":
        ok = re.search(value, code) is None
        return ok, f"expected code not to match regex: {value!r}"

    return True, ""


def _match_spec(code: str, spec: Any, *, negative: bool) -> Tuple[bool, str]:
    """
    Generic spec matcher.

    Supported forms:
    - str
    - dict
    - fallback repr match rules skipped as pass
    """
    if spec is None:
        return True, ""

    if isinstance(spec, str):
        text = spec.strip()
        if not text:
            return True, ""
        if negative:
            ok = text not in code
            return ok, f"forbidden text found: {text!r}"
        ok = text in code
        return ok, f"required text missing: {text!r}"

    if isinstance(spec, dict):
        ok, detail = _match_text_spec(code, spec)
        return ok, detail

    # Unknown spec type: pass through but note in meta later if needed.
    return True, ""


def _build_directive(
    *,
    missing_symbols: Sequence[str],
    missing_calls: Sequence[str],
    forbidden_hits: Sequence[str],
    match_failures: Sequence[str],
) -> Tuple[Directive, ...]:
    directives: List[Directive] = []

    if missing_symbols:
        directives.append(
            Directive(
                action="add_required_symbols",
                payload={"symbols": list(missing_symbols)},
            )
        )

    if missing_calls:
        directives.append(
            Directive(
                action="add_required_calls",
                payload={"calls": list(missing_calls)},
            )
        )

    if forbidden_hits:
        directives.append(
            Directive(
                action="remove_forbidden_patterns",
                payload={"items": list(forbidden_hits)},
            )
        )

    if match_failures:
        directives.append(
            Directive(
                action="satisfy_match_specs",
                payload={"items": list(match_failures)},
            )
        )

    return tuple(directives)


@dataclass
class ConstraintVerifier:
    """
    Minimal verifier for generated code.

    Responsibilities:
    - verify required symbols/calls
    - verify forbidden specs
    - verify match specs
    - return structured violations/directives/coverage

    Non-responsibilities:
    - no execution
    - no AST rebuilding
    - no Beacon reasoning reconstruction
    """
    print_io: bool = False

    def _print(self, message: str) -> None:
        if self.print_io:
            print(f"[ConstraintVerifier] {message}")

    def verify(self, code: str, constraints: Constraints) -> VerifierResult:
        code = _safe_text(code)
        violations: List[Violation] = []

        required_symbols = tuple(constraints.required_symbols or ())
        required_calls = tuple(constraints.required_calls or ())
        forbidden_specs = tuple(constraints.forbidden_specs or ())
        match_specs = tuple(constraints.match_specs or ())

        self._print(
            f"start verify: symbols={len(required_symbols)} "
            f"calls={len(required_calls)} forbidden={len(forbidden_specs)} match={len(match_specs)}"
        )

        missing_symbols: List[str] = []
        for sym in required_symbols:
            if not _contains_symbol(code, sym):
                missing_symbols.append(sym)
                violations.append(
                    Violation(
                        kind="missing_required_symbol",
                        detail=f"required symbol not found: {sym}",
                        spec_ref={"symbol": sym},
                    )
                )

        missing_calls: List[str] = []
        for call_name in required_calls:
            if not _contains_call(code, call_name):
                missing_calls.append(call_name)
                violations.append(
                    Violation(
                        kind="missing_required_call",
                        detail=f"required call not found: {call_name}",
                        spec_ref={"call": call_name},
                    )
                )

        forbidden_hits: List[str] = []
        for spec in forbidden_specs:
            ok, detail = _match_spec(code, spec, negative=True)
            if not ok:
                forbidden_hits.append(detail)
                violations.append(
                    Violation(
                        kind="forbidden_spec_hit",
                        detail=detail,
                        spec_ref={"spec": spec},
                    )
                )

        match_failures: List[str] = []
        for spec in match_specs:
            ok, detail = _match_spec(code, spec, negative=False)
            if not ok:
                match_failures.append(detail)
                violations.append(
                    Violation(
                        kind="match_spec_failed",
                        detail=detail,
                        spec_ref={"spec": spec},
                    )
                )

        coverage: Dict[str, int] = {
            "required_symbols_total": len(required_symbols),
            "required_symbols_hit": len(required_symbols) - len(missing_symbols),
            "required_calls_total": len(required_calls),
            "required_calls_hit": len(required_calls) - len(missing_calls),
            "forbidden_specs_total": len(forbidden_specs),
            "forbidden_specs_passed": len(forbidden_specs) - len(forbidden_hits),
            "match_specs_total": len(match_specs),
            "match_specs_passed": len(match_specs) - len(match_failures),
        }

        ok = len(violations) == 0
        directives = _build_directive(
            missing_symbols=missing_symbols,
            missing_calls=missing_calls,
            forbidden_hits=forbidden_hits,
            match_failures=match_failures,
        )

        self._print(
            f"verify done: ok={ok} violations={len(violations)} directives={len(directives)}"
        )

        return VerifierResult(
            ok=ok,
            coverage=coverage,
            violations=tuple(violations),
            directives=directives,
            meta={
                "verifier": "ConstraintVerifier",
                "mode": "text-pattern",
                "code_len": len(code),
            },
        )


def verify_code(
    code: str,
    constraints: Constraints,
    *,
    print_io: bool = False,
) -> VerifierResult:
    """
    Convenience function for direct use.
    """
    verifier = ConstraintVerifier(print_io=print_io)
    return verifier.verify(code=code, constraints=constraints)