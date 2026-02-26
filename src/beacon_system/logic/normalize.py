from __future__ import annotations

from beacon_system.types import BeaconIR


def normalize_ir(ir: BeaconIR) -> BeaconIR:
    ir.nodes = sorted(ir.nodes, key=lambda n: (n.kind, n.node_id, n.content))
    ir.edges = sorted(ir.edges, key=lambda e: (e.kind, e.src, e.dst))
    ir.symbols = sorted(set(ir.symbols))
    ir.forbidden = sorted(set(ir.forbidden))
    ir.skeleton = [f"{n.kind}:{n.content}" for n in ir.nodes]
    return ir
