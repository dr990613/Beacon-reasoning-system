# scripts/run_logic_smoke.py
# -*- coding: utf-8 -*-
"""
Smoke script for Beacon Logic (logic-only).

What it does:
1) Builds a tiny in-memory "project" with entry() -> helper()
2) Runs logic.engine.build() twice to assert determinism
3) Prints a compact summary to console
4) Saves outputs next to this script:
   - logic_smoke_ir.json
   - logic_smoke_constraints.json
   - logic_smoke_debug.json
   - logic_smoke_determinism.txt

Run:
    python scripts/run_logic_smoke.py
"""

from __future__ import annotations

import json
from pathlib import Path

from .engine import build, ProjectIndex
from .state import ReaderConfig
from .normalize import stable_json
from dataclasses import asdict


def _dump_json(path: Path, obj) -> None:
    # Use stable_json for deterministic & readable output
    path.write_text(stable_json(obj), encoding="utf-8")


def main() -> None:
    # A minimal project:
    # - helper has a validation guard (L-VAL should catch it)
    # - entry calls helper (G-CALL / call edge should appear; required_calls should include helper if meta present)
    src = """
def normalize(x):
    if x is None:
        return 0
    return abs(x)


def compute(a, b):
    temp = normalize(a)
    result = temp + b
    return result


def finalize(v):
    return v * 2


def entry(p, q):
    if p < 0:
        raise ValueError("negative")

    core = compute(p, q)
    output = finalize(core)

    print("done")   # 非语义关键调用
    return output
""".lstrip()

    proj = ProjectIndex(
        entry_file="proj.py",
        entry_qualname="entry",
        files={"proj.py": src},
    )

    task = {"id": "logic_smoke_task"}  # MVP placeholder TaskObject
    cfg = ReaderConfig(enable_global=True, validation_filter=True)

    # Run twice to verify determinism
    r1 = build(task, proj, cfg, seed=0, with_debug=True)
    r2 = build(task, proj, cfg, seed=0, with_debug=True)

    ir1_json = stable_json(r1.ir)
    ir2_json = stable_json(r2.ir)
    cons1_json = stable_json(r1.constraints.to_dict())
    cons2_json = stable_json(r2.constraints.to_dict())

    ir_same = (ir1_json == ir2_json)
    cons_same = (cons1_json == cons2_json)

    # Console summary
    print("=== Beacon Logic Smoke ===")
    print(f"IR nodes: {len(r1.ir.nodes)}")
    print(f"IR edges: {len(r1.ir.edges)}")
    print(f"IR forbidden: {len(r1.ir.forbidden)}")
    print(f"Symbols.calls: {r1.ir.symbols.get('calls', [])}")
    print(f"Constraints.required_calls: {list(r1.constraints.required_calls)}")
    print(f"Determinism (IR): {ir_same}")
    print(f"Determinism (Constraints): {cons_same}")
    if r1.debug:
        print(f"IR hash: {r1.debug.ir_hash}")
        print(f"Constraints hash: {r1.debug.constraints_hash}")

    # Save artifacts next to this script
    out_dir = Path(__file__).resolve().parent
    _dump_json(out_dir / "logic_smoke_ir.json", r1.ir)
    _dump_json(out_dir / "logic_smoke_constraints.json", r1.constraints.to_dict())
    _dump_json(out_dir / "logic_smoke_debug.json", (asdict(r1.debug) if r1.debug else {}))

    (out_dir / "logic_smoke_determinism.txt").write_text(
        "\n".join(
            [
                f"IR deterministic: {ir_same}",
                f"Constraints deterministic: {cons_same}",
                "",
                "If false, inspect stable_json diffs between runs.",
            ]
        ),
        encoding="utf-8",
    )

    print(f"\nSaved outputs to: {out_dir}")
    print("  - logic_smoke_ir.json")
    print("  - logic_smoke_constraints.json")
    print("  - logic_smoke_debug.json")
    print("  - logic_smoke_determinism.txt")


if __name__ == "__main__":
    main()