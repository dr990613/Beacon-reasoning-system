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

from dataclasses import dataclass
from typing import Any, Dict, Protocol, Tuple, runtime_checkable

from ..types import TaskObject, ProjectIndex, ExecutionResult


@runtime_checkable
class TaskAdapter(Protocol):
    """
    TaskAdapter builds the normalized TaskObject + ProjectIndex.

    - Absorb unknown benchmark fields into TaskObject.context/meta.
    - ProjectIndex is read-only for logic (built by adapter).
    """

    def build_task(self) -> Tuple[TaskObject, ProjectIndex]:
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

    patch: a dict describing how to inject generated code into the environment,
           e.g., {"target_file": ..., "target_qualname": ..., "new_code": ...}
    """

    def run(self, task: TaskObject, patch: Dict[str, Any]) -> ExecutionResult:
        ...

    def snapshot(self) -> Dict[str, Any]:
        """
        Return a JSON-serializable snapshot for reproducibility.
        """
        ...