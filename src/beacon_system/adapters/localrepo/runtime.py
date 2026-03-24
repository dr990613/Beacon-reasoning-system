# -*- coding: utf-8 -*-

"""
Local repository runtime adapter.

Responsibilities:
- Execute commands inside a local repository
- Return normalized runtime result only
- No reasoning, no acceptance semantics

Current status:
- Not part of the main logic chain
- Used only when tests / commands need to be executed
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence
import os
import subprocess

from ..base import RuntimeAdapter, RuntimeResult


def _normalize_env(env: Optional[Dict[str, str]]) -> Dict[str, str]:
    """
    Merge user env with current process env.
    """
    merged = dict(os.environ)
    if env:
        for key, value in env.items():
            merged[str(key)] = str(value)
    return merged


class LocalRepoRuntimeAdapter(RuntimeAdapter):
    """
    Minimal subprocess-based runtime adapter.

    Design goals:
    - run commands only
    - normalize outputs
    - avoid hidden reasoning
    """

    def run(
        self,
        *,
        command: Sequence[str],
        cwd: str | Path,
        env: Optional[Dict[str, str]] = None,
        timeout_sec: Optional[int] = None,
    ) -> RuntimeResult:
        """
        Execute command in the given working directory.

        Returns:
            RuntimeResult with stdout/stderr/exit_code only.
        """
        if not command:
            raise ValueError("command must be a non-empty sequence.")

        workdir = Path(cwd).resolve()
        if not workdir.exists():
            raise FileNotFoundError(f"cwd not found: {workdir}")
        if not workdir.is_dir():
            raise NotADirectoryError(f"cwd is not a directory: {workdir}")

        completed = subprocess.run(
            list(command),
            cwd=str(workdir),
            env=_normalize_env(env),
            text=True,
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )

        return RuntimeResult(
            ok=(completed.returncode == 0),
            exit_code=int(completed.returncode),
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
            command=[str(x) for x in command],
            cwd=str(workdir),
        )