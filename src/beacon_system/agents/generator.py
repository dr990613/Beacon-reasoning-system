from __future__ import annotations

from beacon_system.types import BeaconIR, Constraints, VerifierReport


class CodeGenerator:
    def generate(self, task_signature: str, ir: BeaconIR, constraints: Constraints) -> str:
        body = "\n    ".join(["# generated from beacon ir"] + [f"# {line}" for line in ir.skeleton] + ["raise NotImplementedError"])
        return f"def {task_signature}:\n    {body}\n"

    def revise(self, prior_code: str, report: VerifierReport) -> str:
        directives = "\n".join(f"# directive: {d}" for d in report.directives)
        return f"{prior_code}\n{directives}\n"
