from beacon_system.main import Orchestrator
from beacon_system.types import TaskObject


def test_pipeline_smoke():
    task = TaskObject(
        id="smoke",
        lang="python",
        signature="solve(x)",
        doc="return solved x",
        context={"imports": ["import math"]},
    )
    result = Orchestrator().run_task(task)
    assert "ir" in result
    assert "constraints" in result
    assert "code" in result
    assert "verifier" in result
