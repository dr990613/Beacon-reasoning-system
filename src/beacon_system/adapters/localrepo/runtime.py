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
- capture outputs and return ExecResult
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
from ...types import ExecResult, TaskObject
from .patcher import PatchError, apply_patch


@dataclass
class LocalRepoRuntimeAdapter(RuntimeAdapter):
    repo_root: str
    run_cmd: str = "pytest -q"
    work_dir: Optional[str] = None   # if provided, use as base; else system temp dir
    timeout_sec: Optional[int] = None
    print_io: bool = False

    def __post_init__(self) -> None:
        rr = os.path.abspath(self.repo_root)
        if not os.path.isdir(rr):
            raise ValueError(f"repo_root not found: {rr}")
        self.repo_root = rr

        if self.work_dir:
            self.work_dir = os.path.abspath(self.work_dir)

    def _print(self, message: str) -> None:
        if self.print_io:
            print(f"[LocalRepoRuntime] {message}")

    def snapshot(self) -> Dict[str, Any]:
        return {
            "name": "localrepo",
            "repo_root": self.repo_root,
            "run_cmd": self.run_cmd,
            "work_dir": self.work_dir,
            "timeout_sec": self.timeout_sec,
        }

    def run(self, task: TaskObject, patch: Dict[str, Any]) -> ExecResult:
        """
        patch expects:
          - target_file: str
          - target_qualname: str
          - new_code: str
        """
        target_file = str(patch.get("target_file") or task.target.get("file") or "")
        target_qualname = str(patch.get("target_qualname") or task.target.get("qualname") or "")
        new_code = str(patch.get("new_code") or "")

        if not target_file:
            return ExecResult(
                status="error",
                return_code=2,
                stdout="",
                stderr="missing target_file",
                trace="missing target_file",
                metrics={"phase": "runtime-validate"},
            )

        if not target_qualname:
            return ExecResult(
                status="error",
                return_code=2,
                stdout="",
                stderr="missing target_qualname",
                trace="missing target_qualname",
                metrics={"phase": "runtime-validate", "target_file": target_file},
            )

        if not new_code.strip():
            return ExecResult(
                status="error",
                return_code=2,
                stdout="",
                stderr="missing new_code",
                trace="missing new_code",
                metrics={
                    "phase": "runtime-validate",
                    "target_file": target_file,
                    "target_qualname": target_qualname,
                },
            )

        # Prepare isolated working copy
        base = self.work_dir
        if base:
            os.makedirs(base, exist_ok=True)
            work_root = tempfile.mkdtemp(prefix="beacon-run-", dir=base)
        else:
            work_root = tempfile.mkdtemp(prefix="beacon-run-")

        self._print(f"work_root={work_root}")
        self._print("phase=copy")

        try:
            # Copy repo into isolated work root
            shutil.copytree(self.repo_root, work_root, dirs_exist_ok=True)

            self._print("phase=patch")
            try:
                apply_patch(work_root, target_file, target_qualname, new_code)
            except PatchError as e:
                self._print(f"patch failed: {e}")
                return ExecResult(
                    status="error",
                    return_code=2,
                    stdout="",
                    stderr=str(e),
                    trace=str(e),
                    metrics={
                        "phase": "patch",
                        "target_file": target_file,
                        "target_qualname": target_qualname,
                        "work_root": work_root,
                    },
                )

            self._print(f"phase=run cmd={self.run_cmd!r}")
            start = time.time()

            try:
                proc = subprocess.run(
                    self.run_cmd,
                    cwd=work_root,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
                dur = time.time() - start

                stdout = proc.stdout or ""
                stderr = proc.stderr or ""
                trace = (stdout + "\n" + stderr).strip()

                status = "pass" if proc.returncode == 0 else "fail"
                self._print(f"run finished: status={status} return_code={proc.returncode}")

                return ExecResult(
                    status=status,
                    return_code=int(proc.returncode),
                    stdout=stdout,
                    stderr=stderr,
                    trace=trace,
                    metrics={
                        "phase": "run",
                        "duration_s": dur,
                        "cmd": self.run_cmd,
                        "timeout_sec": self.timeout_sec,
                        "target_file": target_file,
                        "target_qualname": target_qualname,
                        "work_root": work_root,
                    },
                )

            except subprocess.TimeoutExpired as e:
                dur = time.time() - start
                stdout = e.stdout or ""
                stderr = e.stderr or ""
                trace = (str(e) + "\n" + (stdout or "") + "\n" + (stderr or "")).strip()

                self._print("run timeout")

                return ExecResult(
                    status="error",
                    return_code=124,
                    stdout=stdout,
                    stderr=stderr or f"command timed out after {self.timeout_sec}s",
                    trace=trace,
                    metrics={
                        "phase": "run",
                        "duration_s": dur,
                        "cmd": self.run_cmd,
                        "timeout_sec": self.timeout_sec,
                        "timed_out": True,
                        "target_file": target_file,
                        "target_qualname": target_qualname,
                        "work_root": work_root,
                    },
                )

            except Exception as e:
                dur = time.time() - start
                self._print(f"run exception: {e}")

                return ExecResult(
                    status="error",
                    return_code=3,
                    stdout="",
                    stderr=str(e),
                    trace=str(e),
                    metrics={
                        "phase": "run",
                        "duration_s": dur,
                        "cmd": self.run_cmd,
                        "timeout_sec": self.timeout_sec,
                        "target_file": target_file,
                        "target_qualname": target_qualname,
                        "work_root": work_root,
                    },
                )

        finally:
            # MVP behavior: always clean up isolated working copy.
            # If later you need keep-on-fail, add a flag instead of changing core behavior here.
            self._print("cleanup work_root")
            shutil.rmtree(work_root, ignore_errors=True)