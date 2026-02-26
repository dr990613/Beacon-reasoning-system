from __future__ import annotations

from beacon_system.logic.matchers import match_spec
from beacon_system.types import Constraints, VerifierReport, Violation


class Verifier:
    def verify(self, code: str, constraints: Constraints) -> VerifierReport:
        violations: list[Violation] = []
        for req in constraints.required:
            text = req.payload.get("content", "")
            if text and text not in code:
                violations.append(Violation("missing_required", f"missing required content: {text}"))
        for spec in constraints.match_spec:
            if not match_spec(spec, code):
                violations.append(Violation("match_spec", f"matcher failed: {spec}"))
        accepted = not violations
        directives = [] if accepted else ["Include missing required beacon semantics in code comments or logic."]
        return VerifierReport(accepted=accepted, violations=violations, directives=directives)
