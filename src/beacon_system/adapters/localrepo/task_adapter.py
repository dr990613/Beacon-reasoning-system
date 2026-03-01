# src/beacon_system/adapters/localrepo/task_adapter.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..base import TaskAdapter
from ...types import TaskObject, ProjectIndex


def _relpath(path: str, root: str) -> str:
    rp = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
    return rp.replace("\\", "/")


@dataclass
class LocalRepoTaskAdapter(TaskAdapter):
    repo_root: str
    target_file: Optional[str]
    target_qualname: Optional[str]
    spec: str = ""
    context: Dict[str, Any] = None
    meta: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.context is None:
            self.context = {}
        if self.meta is None:
            self.meta = {}

        if not self.target_file or not self.target_qualname:
            raise ValueError("localrepo requires target_file and target_qualname")

        rr = os.path.abspath(self.repo_root)
        if not os.path.isdir(rr):
            raise ValueError(f"repo_root not found: {rr}")

        abs_target = os.path.join(rr, self.target_file)
        if not os.path.isfile(abs_target):
            raise ValueError(f"target_file not found: {abs_target}")

        self.repo_root = rr
        self.target_file = _relpath(abs_target, rr)

    def build_task(self) -> Tuple[TaskObject, ProjectIndex]:
        tid = self.meta.get("id") or f"localrepo-{uuid.uuid4().hex[:12]}"

        task = TaskObject(
            id=str(tid),
            lang=str(self.meta.get("lang") or "python"),
            level=str(self.meta.get("level") or "function"),
            target={"file": str(self.target_file), "qualname": str(self.target_qualname)},
            spec=str(self.spec or ""),
            context=dict(self.context or {}),
            meta=dict(self.meta or {}),
        )

        # Align with current logic.engine expectations:
        # - entry is on ProjectIndex
        # - files is dict[file->source], so logic can do project_index.files.items()
        entry_file = str(self.target_file)
        entry_qualname = str(self.target_qualname)

        abs_entry = os.path.join(self.repo_root, entry_file)
        with open(abs_entry, "r", encoding="utf-8") as f:
            entry_source = f.read()

        index = ProjectIndex(
            root=self.repo_root,
            entry_file=entry_file,
            entry_qualname=entry_qualname,
            files={entry_file: entry_source},
            ast_index={},
            symbols={},
            callgraph={},
        )
        return task, index

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": "localrepo",
            "repo_root": self.repo_root,
            "target_file": self.target_file,
            "target_qualname": self.target_qualname,
            "spec_len": len(self.spec or ""),
            "context_keys": sorted(list((self.context or {}).keys())),
            "meta_keys": sorted(list((self.meta or {}).keys())),
        }