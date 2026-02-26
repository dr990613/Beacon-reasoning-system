from __future__ import annotations

from beacon_system.types import BeaconIR, ConstraintRule, Constraints


def compile_constraints(ir: BeaconIR) -> Constraints:
    required = [ConstraintRule("node", {"content": n.content}) for n in ir.nodes if n.kind in {"dependency", "output"}]
    forbidden = [ConstraintRule("pattern", {"content": f}) for f in ir.forbidden]
    match_spec = [{"op": "contains", "value": n.content} for n in ir.nodes if n.kind == "dependency"]
    return Constraints(required=required, forbidden=forbidden, match_spec=match_spec)
