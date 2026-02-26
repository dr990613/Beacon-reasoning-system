from beacon_system.logic.engine import BeaconEngine
from beacon_system.types import TaskObject


def test_engine_is_deterministic():
    task = TaskObject(
        id="t1",
        lang="python",
        signature="foo(x)",
        doc="compute value",
        context={"imports": ["import os", "import os"]},
    )
    engine = BeaconEngine()
    ir1, c1 = engine.run(task)
    ir2, c2 = engine.run(task)
    assert ir1.to_dict() == ir2.to_dict()
    assert [r.payload for r in c1.required] == [r.payload for r in c2.required]
