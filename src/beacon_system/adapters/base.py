# -*- coding: utf-8 -*-

"""
Base adapter contracts.

Responsibilities:
- Define minimal TaskAdapter / RuntimeAdapter interfaces
- Keep adapter boundary explicit
- No Beacon logic here
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import ProjectIndex, TaskObject


@dataclass(frozen=True)
class PatchTarget:
    """
    Minimal patch target description used by local patch infra.
    """
    file_path: str
    target_name: Optional[str] = None
    qualname: Optional[str] = None
    lineno: Optional[int] = None
    end_lineno: Optional[int] = None
    language: Optional[str] = None


@dataclass(frozen=True)
class RuntimeResult:
    """
    Minimal runtime result contract.
    """
    ok: bool
    exit_code: int
    stdout: str
    stderr: str
    command: List[str]
    cwd: Optional[str] = None


class TaskAdapter(ABC):
    """
    Minimal input adapter contract.

    Adapter layer should:
    - read raw task / repo context
    - build TaskObject
    - build ProjectIndex
    - provide enough source/spec context for downstream logic

    Adapter layer should NOT:
    - perform Beacon reasoning
    - inject model logic
    """

    @abstractmethod
    def load_task(self, raw_task: Dict[str, Any]) -> TaskObject:
        """
        Convert a raw benchmark/task record into TaskObject.
        """
        raise NotImplementedError

    @abstractmethod
    def build_project_index(
        self,
        *,
        task: TaskObject,
        project_root: str | Path,
    ) -> ProjectIndex:
        """
        Build a ProjectIndex for the given task and repo root.
        """
        raise NotImplementedError

    @abstractmethod
    def load(
        self,
        *,
        raw_task: Dict[str, Any],
        project_root: str | Path,
    ) -> Tuple[TaskObject, ProjectIndex]:
        """
        Convenience entry:
            raw_task + repo -> (TaskObject, ProjectIndex)
        """
        raise NotImplementedError


class RuntimeAdapter(ABC):
    """
    Minimal execution/runtime adapter contract.

    Runtime layer should:
    - run commands/tests
    - return outputs only

    Runtime layer should NOT:
    - perform reasoning
    - decide acceptance semantics
    """

    @abstractmethod
    def run(
        self,
        *,
        command: Sequence[str],
        cwd: str | Path,
        env: Optional[Dict[str, str]] = None,
        timeout_sec: Optional[int] = None,
    ) -> RuntimeResult:
        raise NotImplementedError