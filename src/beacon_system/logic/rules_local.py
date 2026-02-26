from __future__ import annotations

from beacon_system.logic.anchors import make_anchor
from beacon_system.logic.state import ReasoningState


def apply_local_rules(state: ReasoningState, signature: str, doc: str) -> None:
    out_id = make_anchor(state.task_id, "output")
    dep_id = make_anchor(state.task_id, "core_logic")
    state.ir.nodes.extend(
        [
            __import__("beacon_system.types", fromlist=["BeaconNode"]).BeaconNode(out_id, "output", signature),
            __import__("beacon_system.types", fromlist=["BeaconNode"]).BeaconNode(dep_id, "dependency", doc.strip() or "implement task"),
        ]
    )
    state.ir.edges.append(__import__("beacon_system.types", fromlist=["BeaconEdge"]).BeaconEdge(dep_id, out_id, "dataflow"))
    state.ir.provenance[out_id] = ["L-OUT"]
    state.ir.provenance[dep_id] = ["L-DEP"]
