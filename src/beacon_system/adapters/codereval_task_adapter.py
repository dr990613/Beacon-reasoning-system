"""
# src/beacon_system/adapters/codereval_task_adapter.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import inspect
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .base import TaskAdapter
from ..types import TaskObject, ProjectIndex


def _pick(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _normalize_path_like(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    return str(value).replace("\\", "/").strip()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _iter_python_files(root: Path) -> List[str]:
    out: List[str] = []
    for p in root.rglob("*.py"):
        if ".git" in p.parts:
            continue
        try:
            out.append(str(p.relative_to(root)).replace("\\", "/"))
        except Exception:
            out.append(str(p))
    out.sort()
    return out


def _safe_parse(code: str) -> Optional[ast.AST]:
    try:
        return ast.parse(code)
    except Exception:
        return None


def _infer_signature_from_source(source: str, qualname: str) -> Optional[str]:
    tree = _safe_parse(source)
    if tree is None:
        return None

    parts = qualname.split(".")
    if len(parts) == 1:
        fn_name = parts[0]
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == fn_name:
                seg = ast.get_source_segment(source, node)
                if seg:
                    first = seg.strip().splitlines()[0]
                    return first.rstrip(":") + ":"
        return None

    if len(parts) == 2:
        cls_name, fn_name = parts
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == cls_name:
                for sub in node.body:
                    if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub.name == fn_name:
                        seg = ast.get_source_segment(source, sub)
                        if seg:
                            first = seg.strip().splitlines()[0]
                            return first.rstrip(":") + ":"
        return None

    return None


def _infer_target_file(raw_task: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return _normalize_path_like(override)  # type: ignore[return-value]

    candidates = [
        _pick(raw_task, "target_file", "file", "file_path", "path"),
        _pick(raw_task.get("target") or {}, "file", "path", "file_path"),
        _pick(raw_task.get("metadata") or {}, "target_file", "file", "path"),
    ]
    for c in candidates:
        if c:
            return _normalize_path_like(str(c))  # type: ignore[return-value]

    raise ValueError("Cannot infer target_file from task. Please pass override_target_file.")


def _infer_target_qualname(raw_task: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return str(override).strip()

    candidates = [
        _pick(raw_task, "target_qualname", "qualname", "function_name", "method_name", "symbol", "name"),
        _pick(raw_task.get("target") or {}, "qualname", "symbol", "function_name", "method_name"),
        _pick(raw_task.get("metadata") or {}, "target_qualname", "qualname", "symbol"),
    ]
    for c in candidates:
        if c:
            return str(c).strip()

    raise ValueError("Cannot infer target_qualname from task. Please pass override_target_qualname.")


def _infer_spec_text(raw_task: Dict[str, Any]) -> str:
    pieces: List[str] = []

    for key in ("prompt", "instruction", "description", "docstring", "spec", "task_description"):
        v = raw_task.get(key)
        if isinstance(v, str) and v.strip():
            pieces.append(v.strip())

    meta = raw_task.get("metadata") or {}
    if isinstance(meta, dict):
        for key in ("prompt", "instruction", "description", "docstring", "spec"):
            v = meta.get(key)
            if isinstance(v, str) and v.strip():
                pieces.append(v.strip())

    if not pieces:
        return "Implement the target function/method so that the benchmark-aligned tests pass."

    return "\n\n".join(dict.fromkeys(pieces))


def _resolve_target_file(project_root: Path, target_file_rel: str) -> Path:
    candidates = [
        project_root / target_file_rel,
        project_root / "src" / target_file_rel,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        "Target file not found under project root. Tried:\n"
        + "\n".join(str(x) for x in candidates)
    )


def _make_context_blocks(
    *,
    raw_task: Dict[str, Any],
    source_text: str,
    target_file_rel: str,
    target_qualname: str,
) -> List[str]:
    blocks: List[str] = [
        "You are patching a real CoderEval task in an existing codebase.",
        "Return only the target function or method implementation.",
        "Do not rewrite the whole file.",
        "Keep surrounding project contracts stable.",
        f"Target file: {target_file_rel}",
        f"Target qualname: {target_qualname}",
        "Existing source context:",
        source_text,
    ]

    all_context = raw_task.get("all_context")
    if isinstance(all_context, str) and all_context.strip():
        blocks.append("Additional benchmark context:")
        blocks.append(all_context.strip())

    dependency = raw_task.get("dependency")
    if isinstance(dependency, str) and dependency.strip():
        blocks.append("Dependency info:")
        blocks.append(dependency.strip())

    return blocks


def _construct_type(cls: Any, preferred: Dict[str, Any]) -> Any:
    """
    Introspective constructor to reduce dependency on exact dataclass signature.
    """
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


class CoderEvalTaskAdapter(TaskAdapter):
    """
    Formal TaskAdapter for CoderEval.

    Responsibilities:
    - translate one raw benchmark task into TaskObject
    - build a minimal but useful ProjectIndex
    - provide a stable snapshot for artifacts/debugging
    """

    def __init__(
        self,
        *,
        raw_task: Dict[str, Any],
        json_path: Path,
        project_root: Path,
        override_target_file: Optional[str] = None,
        override_target_qualname: Optional[str] = None,
        level: str = "project_runnable",
    ):
        self._raw_task = dict(raw_task)
        self._json_path = Path(json_path)
        self._project_root = Path(project_root).resolve()
        self._override_target_file = override_target_file
        self._override_target_qualname = override_target_qualname
        self._level = level

        if not self._project_root.exists():
            raise FileNotFoundError(f"project_root not found: {self._project_root}")

    def snapshot(self) -> Dict[str, Any]:
        task_id = str(
            self._raw_task.get("task_id")
            or self._raw_task.get("id")
            or self._raw_task.get("_id")
            or ""
        )
        return {
            "kind": "CoderEvalTaskAdapter",
            "json_path": str(self._json_path),
            "project_root": str(self._project_root),
            "task_id": task_id,
            "override_target_file": self._override_target_file,
            "override_target_qualname": self._override_target_qualname,
            "level": self._level,
        }

    def build_task(self) -> Tuple[TaskObject, ProjectIndex]:
        target_file_rel = _infer_target_file(self._raw_task, self._override_target_file)
        target_qualname = _infer_target_qualname(self._raw_task, self._override_target_qualname)
        target_file_abs = _resolve_target_file(self._project_root, target_file_rel)

        source_text = _read_text(target_file_abs)
        signature = _infer_signature_from_source(source_text, target_qualname)
        if not signature:
            signature = f"def {target_qualname.split('.')[-1]}(...):"

        task_id = str(
            self._raw_task.get("task_id")
            or self._raw_task.get("id")
            or self._raw_task.get("_id")
            or "codereval_task"
        )

        spec_text = _infer_spec_text(self._raw_task)
        context_blocks = _make_context_blocks(
            raw_task=self._raw_task,
            source_text=source_text,
            target_file_rel=target_file_rel,
            target_qualname=target_qualname,
        )

        task_preferred = {
            "id": task_id,
            "task_id": task_id,
            "lang": "python",
            "level": self._level,
            "target": {
                "file": target_file_rel,
                "qualname": target_qualname,
            },
            "spec": spec_text,
            "context": {
                "signature": signature,
                "context_blocks": context_blocks,
                "project_root": str(self._project_root),
                "target_file_abs": str(target_file_abs),
            },
            "meta": {
                "benchmark": "CoderEval",
                "json_path": str(self._json_path),
                "raw_task_id": task_id,
                "target_qualname": target_qualname,
                "target_file": target_file_rel,
                "entry_function_name": target_qualname.split(".")[-1],
                "signature": signature,
                "raw_task": self._raw_task,
            },
        }
        task_obj = _construct_type(TaskObject, task_preferred)

        python_files = _iter_python_files(self._project_root)
        project_index_preferred = {
            "root": str(self._project_root),
            "project_root": str(self._project_root),
            "repo_root": str(self._project_root),
            "files": python_files,
            "file_map": {p: None for p in python_files},
            "meta": {
                "benchmark": "CoderEval",
                "project_root": str(self._project_root),
                "target_file": target_file_rel,
                "target_qualname": target_qualname,
                "python_file_count": len(python_files),
            },
        }
        project_index = _construct_type(ProjectIndex, project_index_preferred)

        return task_obj, project_index