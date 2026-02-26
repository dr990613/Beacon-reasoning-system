from __future__ import annotations

from beacon_system.types import TaskObject


def validate_task(task: TaskObject) -> TaskObject:
    if not task.id.strip():
        raise ValueError("task id cannot be empty")
    if task.lang not in {"python"}:
        raise ValueError(f"unsupported language: {task.lang}")
    if not task.signature.strip():
        raise ValueError("signature cannot be empty")
    return task
