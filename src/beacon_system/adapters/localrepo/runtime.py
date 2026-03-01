# src/beacon_system/adapters/localrepo/runtime.py
# -*- coding: utf-8 -*-

"""
LocalRepo RuntimeAdapter

- copy -> patch -> run command -> collect stdout/stderr/trace
- designed for reproducibility and debuggability (working dir is isolated)

Minimal behavior:
- create a temp working copy under outputs/tmp/<run_id>/ (or system temp)
- apply patch (target file + qualname)
- run command (default: pytest -q)
- capture outputs and return ExecutionResult
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..base import RuntimeAdapter
from ...types import ExecutionResult, TaskObject
from .patcher import apply_patch, PatchError


@dataclass
class LocalRepoRuntimeAdapter(RuntimeAdapter):
    repo_root: str
    run_cmd: str = "pytest -q"
    work_dir: Optional[str] = None  # if provided, use as base; else tempdir

    def __post_init__(self) -> None:
        rr = os.path.abspath(self.repo_root)
        if not os.path.isdir(rr):
            raise ValueError(f"repo_root not found: {rr}")
        self.repo_root = rr

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": "localrepo",
            "repo_root": self.repo_root,
            "run_cmd": self.run_cmd,
            "work_dir": self.work_dir,
        }

    def run(self, task: TaskObject, patch: Dict[str, Any]) -> ExecutionResult:
        """
        patch expects:
          - target_file: str
          - target_qualname: str
          - new_code: str
        """
        target_file = str(patch.get("target_file") or task.target.get("file") or "")
        target_qualname = str(patch.get("target_qualname") or task.target.get("qualname") or "")
        new_code = str(patch.get("new_code") or "")

        # Prepare isolated working copy
        base = self.work_dir
        if base:
            os.makedirs(base, exist_ok=True)
            work_root = tempfile.mkdtemp(prefix="beacon-run-", dir=base)
        else:
            work_root = tempfile.mkdtemp(prefix="beacon-run-")

        try:
            # Copy repo
            shutil.copytree(self.repo_root, work_root, dirs_exist_ok=True)

            # Patch
            try:
                apply_patch(work_root, target_file, target_qualname, new_code)
            except PatchError as e:
                return ExecutionResult(
                    status="error",
                    return_code=2,
                    stdout="",
                    stderr=str(e),
                    trace=str(e),
                    metrics={"phase": "patch"},
                )

            # Run
            start = time.time()
            proc = subprocess.run(
                self.run_cmd,
                cwd=work_root,
                shell=True,
                capture_output=True,
                text=True,
            )
            dur = time.time() - start

            stdout = proc.stdout or ""
            stderr = proc.stderr or ""
            trace = (stdout + "\n" + stderr).strip()

            status = "pass" if proc.returncode == 0 else "fail"
            return ExecutionResult(
                status=status,
                return_code=int(proc.returncode),
                stdout=stdout,
                stderr=stderr,
                trace=trace,
                metrics={"duration_s": dur, "cmd": self.run_cmd},
            )
        finally:
            # Keep work_root for debugging? MVP cleans up. If you want keep-on-fail, add a flag later.
            shutil.rmtree(work_root, ignore_errors=True)