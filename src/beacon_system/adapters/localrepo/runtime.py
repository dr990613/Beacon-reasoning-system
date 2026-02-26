from __future__ import annotations

import subprocess

from beacon_system.adapters.base import RuntimeAdapter
from beacon_system.adapters.localrepo.patcher import replace_function
from beacon_system.types import ExecutionResult


class LocalRepoRuntimeAdapter(RuntimeAdapter):
    def execute(self, patch_code: str, **kwargs) -> ExecutionResult:
        target_file = kwargs["target_file"]
        function_name = kwargs["function_name"]
        command = kwargs.get("command", "pytest -q")
        cwd = kwargs.get("cwd")
        replace_function(target_file, function_name, patch_code)
        proc = subprocess.run(command, cwd=cwd, shell=True, capture_output=True, text=True)
        return ExecutionResult(
            success=proc.returncode == 0,
            command=command,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
