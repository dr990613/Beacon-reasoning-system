"""
# src/beacon_system/adapters/codereval_docker_runtime_adapter.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import inspect
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .base import RuntimeAdapter
from ..types import ExecutionResult


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _extract_code_block(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _extract_target_definition(full_code: str, qualname: str) -> str:
    """
    Extract only the target def from generated content.
    Supports:
    - top-level function: func
    - class method: Class.method
    """
    code = _extract_code_block(full_code)
    parts = qualname.split(".")
    target_name = parts[-1]

    try:
        tree = ast.parse(code)
        if len(parts) == 1:
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name:
                    return ast.unparse(node)

        elif len(parts) == 2:
            cls_name, fn_name = parts
            for node in tree.body:
                if isinstance(node, ast.ClassDef) and node.name == cls_name:
                    for sub in node.body:
                        if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == fn_name:
                            return ast.unparse(sub)

            # allow top-level def fallback for method target
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name:
                    return ast.unparse(node)
    except Exception:
        pass

    return code.strip()


def _locate_def_span(source: str, qualname: str) -> Tuple[int, int, int]:
    """
    Return:
    - start line index (0-based, inclusive)
    - end line index (0-based, exclusive)
    - indent spaces to apply to replacement
    """
    tree = ast.parse(source)
    parts = qualname.split(".")

    if len(parts) == 1:
        fn_name = parts[0]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
                if node.end_lineno is None:
                    raise RuntimeError("AST node missing end_lineno")
                return node.lineno - 1, node.end_lineno, 0

    elif len(parts) == 2:
        cls_name, fn_name = parts
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == fn_name:
                        if sub.end_lineno is None:
                            raise RuntimeError("AST node missing end_lineno")
                        return sub.lineno - 1, sub.end_lineno, sub.col_offset

    raise RuntimeError(f"Cannot locate target definition span for {qualname}")


def _indent_block(text: str, spaces: int) -> str:
    if spaces <= 0:
        return text.strip() + "\n"

    prefix = " " * spaces
    lines = text.strip().splitlines()
    return "\n".join(prefix + line if line.strip() else "" for line in lines) + "\n"


def _patch_source_with_generated_def(source: str, generated_def: str, qualname: str) -> str:
    start, end, indent = _locate_def_span(source, qualname)
    lines = source.splitlines(keepends=True)
    replacement = _indent_block(generated_def, indent)
    return "".join(lines[:start]) + replacement + "".join(lines[end:])


def _construct_type(cls: Any, preferred: Dict[str, Any]) -> Any:
    try:
        sig = inspect.signature(cls)
        kwargs: Dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if name in preferred:
                kwargs[name] = preferred[name]
        return cls(**kwargs)
    except Exception:
        pass

    try:
        obj = cls()
        for k, v in preferred.items():
            try:
                setattr(obj, k, v)
            except Exception:
                pass
        return obj
    except Exception as e:
        raise RuntimeError(f"Failed to construct {getattr(cls, '__name__', repr(cls))}: {e}") from e


class CoderEvalDockerRuntimeAdapter(RuntimeAdapter):
    """
    Formal RuntimeAdapter for CoderEval.

    Strategy:
    - copy project repo to a task-local work directory
    - patch the target function/method with generated code
    - validate patched file syntax
    - run benchmark-aligned evaluation command in Docker
    - return ExecutionResult
    """

    def __init__(
        self,
        *,
        project_root: Path,
        docker_image: Optional[str],
        eval_cmd: Optional[str],
        task_result_dir: Path,
        docker_workdir: str = "/workspace",
        keep_workdir: bool = True,
    ):
        self._project_root = Path(project_root).resolve()
        self._docker_image = docker_image
        self._eval_cmd = eval_cmd or "pytest -q"
        self._task_result_dir = Path(task_result_dir)
        self._docker_workdir = docker_workdir
        self._keep_workdir = keep_workdir

        if not self._project_root.exists():
            raise FileNotFoundError(f"project_root not found: {self._project_root}")

        if not self._docker_image:
            raise ValueError(
                "docker_image is required for formal CoderEval runtime evaluation."
            )

    def snapshot(self) -> Dict[str, Any]:
        return {
            "kind": "CoderEvalDockerRuntimeAdapter",
            "project_root": str(self._project_root),
            "docker_image": self._docker_image,
            "eval_cmd": self._eval_cmd,
            "task_result_dir": str(self._task_result_dir),
            "docker_workdir": self._docker_workdir,
            "keep_workdir": self._keep_workdir,
        }

    def run(self, task: Any, patch: Dict[str, Any]) -> ExecutionResult:
        target_file_rel = str(patch.get("target_file") or "").replace("\\", "/").strip()
        target_qualname = str(patch.get("target_qualname") or "").strip()
        new_code = str(patch.get("new_code") or "")

        if not target_file_rel:
            raise ValueError("Runtime patch missing target_file.")
        if not target_qualname:
            raise ValueError("Runtime patch missing target_qualname.")
        if not new_code.strip():
            raise ValueError("Runtime patch missing new_code.")

        ts = time.strftime("%Y%m%d_%H%M%S")
        work_root = self._task_result_dir / f"work_repo_{ts}"
        if work_root.exists():
            shutil.rmtree(work_root)

        shutil.copytree(self._project_root, work_root)

        work_target = work_root / target_file_rel
        if not work_target.exists():
            alt = work_root / "src" / target_file_rel
            if alt.exists():
                work_target = alt
            else:
                raise FileNotFoundError(f"Patched target file not found in work repo: {target_file_rel}")

        original_source = _read_text(work_target)
        generated_def = _extract_target_definition(new_code, target_qualname)
        patched_source = _patch_source_with_generated_def(
            source=original_source,
            generated_def=generated_def,
            qualname=target_qualname,
        )

        # full-file syntax gate before docker eval
        ast.parse(patched_source)

        _write_text(work_target, patched_source)
        _write_text(self._task_result_dir / "patched_target_file.py", patched_source)
        _write_text(self._task_result_dir / "generated_target_definition.py", generated_def)

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{str(work_root)}:{self._docker_workdir}",
            "-w",
            self._docker_workdir,
            self._docker_image,
            "bash",
            "-lc",
            self._eval_cmd,
        ]

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
        )

        runtime_payload = {
            "docker_cmd": cmd,
            "docker_image": self._docker_image,
            "eval_cmd": self._eval_cmd,
            "work_root": str(work_root),
            "patched_target_file": str(work_target),
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        _write_json(self._task_result_dir / "docker_eval_result.json", runtime_payload)

        status = "pass" if proc.returncode == 0 else "fail"

        preferred = {
            "status": status,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "meta": {
                "docker_image": self._docker_image,
                "eval_cmd": self._eval_cmd,
                "work_root": str(work_root),
                "target_file": target_file_rel,
                "target_qualname": target_qualname,
            },
            "details": runtime_payload,
        }
        return _construct_type(ExecutionResult, preferred)