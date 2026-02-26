from pathlib import Path

from beacon_system.adapters.localrepo.patcher import replace_function
from beacon_system.adapters.localrepo.task_adapter import LocalRepoTaskAdapter


def test_localrepo_task_and_patcher(tmp_path: Path):
    f = tmp_path / "sample.py"
    f.write_text("def add(a, b):\n    return a + b\n")
    task = LocalRepoTaskAdapter().build_task(
        task_id="sample", file_path=str(f), signature="add(a, b)", doc="add two numbers"
    )
    assert task.id == "sample"
    replace_function(str(f), "add", "def add(a, b):\n    return a - b")
    assert "a - b" in f.read_text()
