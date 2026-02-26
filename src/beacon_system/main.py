from __future__ import annotations

from dataclasses import asdict

from beacon_system.adapters.registry import get_adapter
from beacon_system.agents.generator import CodeGenerator
from beacon_system.agents.verifier import Verifier
from beacon_system.logic.engine import BeaconEngine
from beacon_system.memory.memory import MemoryManager
from beacon_system.types import TaskObject


class Orchestrator:
    def __init__(self, memory: MemoryManager | None = None):
        self.engine = BeaconEngine()
        self.generator = CodeGenerator()
        self.verifier = Verifier()
        self.memory = memory or MemoryManager()

    def run_task(self, task: TaskObject) -> dict:
        ir, constraints = self.engine.run(task)
        code = self.generator.generate(task.signature, ir, constraints)
        report = self.verifier.verify(code, constraints)
        if not report.accepted:
            code = self.generator.revise(code, report)
            report = self.verifier.verify(code, constraints)
        return {
            "task": asdict(task),
            "ir": ir.to_dict(),
            "constraints": {
                "required": [asdict(x) for x in constraints.required],
                "forbidden": [asdict(x) for x in constraints.forbidden],
                "match_spec": constraints.match_spec,
            },
            "code": code,
            "verifier": {
                "accepted": report.accepted,
                "violations": [asdict(v) for v in report.violations],
                "directives": report.directives,
            },
        }


def build_localrepo_task(**kwargs):
    adapter_cls, _ = get_adapter("localrepo")
    return adapter_cls().build_task(**kwargs)
