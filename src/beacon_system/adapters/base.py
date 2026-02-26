from __future__ import annotations

from abc import ABC, abstractmethod

from beacon_system.types import ExecutionResult, TaskObject


class TaskAdapter(ABC):
    @abstractmethod
    def build_task(self, **kwargs) -> TaskObject: ...


class RuntimeAdapter(ABC):
    @abstractmethod
    def execute(self, patch_code: str, **kwargs) -> ExecutionResult: ...
