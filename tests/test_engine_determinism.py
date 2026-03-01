# tests/test_engine_determinism.py
# -*- coding: utf-8 -*-

import hashlib
import difflib

from beacon_system.logic.engine import build as logic_build
from beacon_system.io import stable_json
from beacon_system.types import TaskObject, ProjectIndex, ReaderConfig


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _preview(s: str, n: int = 600) -> str:
    s = s or ""
    return s[:n] + ("\n...<truncated>\n" if len(s) > n else "")


def _print_diff(a: str, b: str, title: str) -> None:
    diff = difflib.unified_diff(a.splitlines(), b.splitlines(), fromfile="run1", tofile="run2", lineterm="")
    print(f"\n==== {title} DIFF (first 200 lines) ====")
    for i, line in enumerate(diff):
        if i >= 200:
            print("...<diff truncated>...")
            break
        print(line)


def test_engine_build_is_deterministic_bitwise_with_observability():
    """
    Same (task, project_index, config, seed) -> bitwise identical stable_json for IR/Constraints.
    Run: pytest -q -s tests/test_engine_determinism.py
    """
    dummy_source = (
        "def foo(x):\n"
        "    return x + 1\n"
    )

    task = TaskObject(
        id="determinism-smoke",
        lang="python",
        level="function",
        target={"file": "dummy.py", "qualname": "foo"},
        spec="Implement foo(x) that returns x + 1",
        context={},
        meta={},
    )

    project_index = ProjectIndex(
        root=".",
        entry_file="dummy.py",
        entry_qualname="foo",
        files={"dummy.py": dummy_source},
        ast_index={},
        symbols={},
        callgraph={},
    )

    cfg = ReaderConfig(
        enable_global=True,
        validation_filter=True,
        max_local_nodes=200,
        max_global_inline=None,
    )
    seed = 0

    r1 = logic_build(task=task, project_index=project_index, config=cfg, seed=seed, with_debug=False)
    r2 = logic_build(task=task, project_index=project_index, config=cfg, seed=seed, with_debug=False)

    ir1 = stable_json(r1.ir)
    ir2 = stable_json(r2.ir)
    c1 = stable_json(r1.constraints)
    c2 = stable_json(r2.constraints)

    ir1_h, ir2_h = _sha256(ir1), _sha256(ir2)
    c1_h, c2_h = _sha256(c1), _sha256(c2)

    print("\n==== Determinism Report ====")
    print(f"IR hash   : {ir1_h}  ==  {ir2_h}  -> {ir1_h == ir2_h}")
    print(f"IR length : {len(ir1)} / {len(ir2)}")
    print(f"C hash    : {c1_h}  ==  {c2_h}  -> {c1_h == c2_h}")
    print(f"C length  : {len(c1)} / {len(c2)}")

    if ir1 != ir2:
        print("\n[IR mismatch] preview run1:\n", _preview(ir1))
        print("\n[IR mismatch] preview run2:\n", _preview(ir2))
        _print_diff(ir1, ir2, "BeaconIR stable_json")
    if c1 != c2:
        print("\n[Constraints mismatch] preview run1:\n", _preview(c1))
        print("\n[Constraints mismatch] preview run2:\n", _preview(c2))
        _print_diff(c1, c2, "Constraints stable_json")

    assert ir1 == ir2, f"BeaconIR stable_json mismatch: {ir1_h} != {ir2_h}"
    assert c1 == c2, f"Constraints stable_json mismatch: {c1_h} != {c2_h}"