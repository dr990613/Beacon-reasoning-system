from __future__ import annotations

from pathlib import Path

from beacon_system.adapters.base import TaskAdapter
from beacon_system.task.context import assemble_context
from beacon_system.task.task_object import validate_task
from beacon_system.types import TaskObject


class LocalRepoTaskAdapter(TaskAdapter):
    def build_task(self, *, task_id: str, file_path: str, signature: str, doc: str) -> TaskObject:
        source = Path(file_path).read_text()
        ctx = assemble_context(file_path, source)
        return validate_task(TaskObject(id=task_id, lang="python", signature=signature, doc=doc, context=ctx))
