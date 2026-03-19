# src/beacon_system/adapters/base.py
# -*- coding: utf-8 -*-

"""
Adapter interfaces (Adapter-first)

Hard rules:
- Pipeline depends ONLY on these interfaces.
- Benchmark-specific logic must live in adapters/<name>/.
- No registry here, no env access, no business logic.
"""

from __future__ import annotations

from typing import Any, Dict, Protocol, Tuple, runtime_checkable

from ..types import ExecResult, ProjectIndex, TaskObject


@runtime_checkable
class TaskAdapter(Protocol):
    """
    TaskAdapter builds the normalized TaskObject + ProjectIndex.

    Rules:
    - Absorb unknown benchmark fields into TaskObject.context/meta.
    - ProjectIndex is built by adapter and treated as read-only downstream.
    - Do not run reasoning here.
    """

    def build_task(self) -> Tuple[TaskObject, ProjectIndex]:
        """
        Return:
            (task, project_index)
        """
        ...

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable snapshot for reproducibility.
        """
        ...


@runtime_checkable
class RuntimeAdapter(Protocol):
    """
    RuntimeAdapter runs the task in its benchmark/runtime environment.

    patch:
        A JSON-serializable dict describing how to inject generated code into
        the runtime environment, for example:
        {
            "target_file": ...,
            "target_qualname": ...,
            "new_code": ...
        }

    Notes:
    - Keep this minimal for now.
    - Concrete adapters may internally convert patch dict into stronger objects.
    """

    def run(self, task: TaskObject, patch: Dict[str, Any]) -> ExecResult:
        """
        Execute patched task in the target runtime and return normalized result.
        """
        ...

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable snapshot for reproducibility.
        """
        ...