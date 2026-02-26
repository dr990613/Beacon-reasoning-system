from __future__ import annotations

from beacon_system.logic.anchors import make_anchor
from beacon_system.logic.state import ReasoningState
from beacon_system.types import BeaconEdge, BeaconNode


def apply_global_rules(state: ReasoningState, imports: list[str]) -> None:
    for idx, imp in enumerate(sorted(set(imports))):
        node_id = make_anchor(state.task_id, f"global_{idx}")
        state.ir.nodes.append(BeaconNode(node_id, "global", imp))
        state.ir.provenance[node_id] = ["G-GLOB"]
        state.ir.edges.append(BeaconEdge(node_id, make_anchor(state.task_id, "output"), "context"))
