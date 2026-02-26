from __future__ import annotations

from beacon_system.logic.constraints import compile_constraints
from beacon_system.logic.normalize import normalize_ir
from beacon_system.logic.rules_global import apply_global_rules
from beacon_system.logic.rules_local import apply_local_rules
from beacon_system.logic.state import ReasoningState
from beacon_system.types import Constraints, TaskObject


class BeaconEngine:
    def run(self, task: TaskObject) -> tuple:
        state = ReasoningState(task_id=task.id)
        apply_local_rules(state, task.signature, task.doc)
        apply_global_rules(state, task.context.get("imports", []))
        ir = normalize_ir(state.ir)
        constraints: Constraints = compile_constraints(ir)
        return ir, constraints
