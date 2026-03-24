# -*- coding: utf-8 -*-

"""
Local-repo task adapter.

Responsibilities:
- Read raw benchmark task + local repo
- Build TaskObject + ProjectIndex
- Preserve file context as completely as possible
- Keep target file / qualname / spec / relevant source blocks available

Notes:
- No Beacon logic here
- No model logic here
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json

from ..base import TaskAdapter
from ...types import ProjectIndex, TaskObject


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8", errors="replace")


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_lang(raw: Dict[str, Any]) -> Optional[str]:
    for key in ("lang", "language"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            val = value.strip().lower()
            if val in {"py", "python"}:
                return "python"
            if val in {"java"}:
                return "java"
            return val

    file_path = raw.get("file_path") or raw.get("target_file")
    if isinstance(file_path, str):
        suffix = Path(file_path).suffix.lower()
        if suffix == ".py":
            return "python"
        if suffix == ".java":
            return "java"
    return None


def _guess_target_name(raw: Dict[str, Any]) -> Optional[str]:
    for key in ("target_name", "target_function", "function_name", "entry_function", "name"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _guess_target_file(raw: Dict[str, Any]) -> Optional[str]:
    for key in ("target_file", "file_path", "file_name"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _guess_qualname(raw: Dict[str, Any]) -> Optional[str]:
    qualname = raw.get("qualname")
    if isinstance(qualname, str) and qualname.strip():
        return qualname.strip()

    class_name = raw.get("class_name")
    target_name = _guess_target_name(raw)
    if isinstance(class_name, str) and class_name.strip() and target_name:
        return f"{class_name.strip()}.{target_name}"
    return target_name


def _build_source_text(raw: Dict[str, Any]) -> str:
    """
    Unified source view:
    prefer full file, then code, then empty string.
    """
    file_content = raw.get("file_content")
    if isinstance(file_content, str) and file_content.strip():
        return file_content

    code = raw.get("code")
    if isinstance(code, str) and code.strip():
        return code

    return ""


def _build_context_text(raw: Dict[str, Any]) -> str:
    parts: List[str] = []

    all_context = raw.get("all_context")
    if isinstance(all_context, str) and all_context.strip():
        parts.append(all_context.strip())

    oracle_context = raw.get("oracle_context")
    if isinstance(oracle_context, str) and oracle_context.strip():
        parts.append(oracle_context.strip())
    elif isinstance(oracle_context, dict) and oracle_context:
        parts.append(json.dumps(oracle_context, ensure_ascii=False, indent=2, sort_keys=True))

    dependency = raw.get("dependency")
    if isinstance(dependency, str) and dependency.strip():
        parts.append(dependency.strip())

    docstring = raw.get("docstring")
    if isinstance(docstring, str) and docstring.strip():
        parts.append(docstring.strip())

    return "\n\n".join(parts).strip()


def _build_location_span(raw: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    lineno = _safe_int(raw.get("lineno"))
    end_lineno = _safe_int(raw.get("end_lineno"))
    if lineno is None or end_lineno is None:
        return None
    if lineno <= 0 or end_lineno < lineno:
        return None
    return (lineno, end_lineno)


def _task_kwargs_from_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map raw benchmark fields into TaskObject contract.

    This function assumes TaskObject already contains the standardized
    fields discussed in your spec, plus original-compatibility fields.
    """
    task_id = raw.get("task_id") or raw.get("_id")
    target_name = _guess_target_name(raw)
    target_file = _guess_target_file(raw)
    qualname = _guess_qualname(raw)
    source_text = _build_source_text(raw)
    context_text = _build_context_text(raw)
    location_span = _build_location_span(raw)

    kwargs: Dict[str, Any] = {
        # raw-compatible fields
        "task_id": task_id,
        "name": raw.get("name"),
        "docstring": raw.get("docstring"),
        "code": raw.get("code"),
        "file_content": raw.get("file_content"),
        "all_context": raw.get("all_context"),
        "file_path": raw.get("file_path"),
        "file_name": raw.get("file_name"),
        "class_name": raw.get("class_name"),
        "project": raw.get("project"),
        "package": raw.get("package"),
        "level": raw.get("level"),
        "lineno": _safe_int(raw.get("lineno")),
        "end_lineno": _safe_int(raw.get("end_lineno")),
        "oracle_context": raw.get("oracle_context"),
        "dependency": raw.get("dependency"),
        "human_label": raw.get("human_label"),
        # standardized fields
        "lang": _normalize_lang(raw),
        "target_name": target_name,
        "target_file": target_file,
        "qualname": qualname,
        # helper views
        "source_text": source_text,
        "context_text": context_text,
        "location_span": location_span,
    }

    # keep optional commonly-used normalized fields if your TaskObject defines them
    optional_fields = {
        "signature": raw.get("signature"),
        "target_function": raw.get("target_function") or target_name,
        "entry_function": raw.get("entry_function"),
        "instruction": raw.get("instruction"),
        "prompt": raw.get("prompt"),
        "context_blocks": raw.get("context_blocks"),
        "runnable_level": raw.get("runnable_level") or raw.get("level"),
    }
    kwargs.update(optional_fields)
    return kwargs


def _dataclass_field_names(cls: type) -> set[str]:
    if is_dataclass(cls):
        return {f.name for f in fields(cls)}
    return set()


def _construct_task_object(raw: Dict[str, Any]) -> TaskObject:
    """
    Construct TaskObject in a schema-tolerant way:
    only pass fields that actually exist on TaskObject.
    """
    kwargs = _task_kwargs_from_raw(raw)
    allowed = _dataclass_field_names(TaskObject)
    if allowed:
        kwargs = {k: v for k, v in kwargs.items() if k in allowed}
    return TaskObject(**kwargs)  # type: ignore[arg-type]


def _iter_repo_files(project_root: Path, lang: Optional[str]) -> List[Path]:
    suffixes: List[str]
    if lang == "python":
        suffixes = [".py"]
    elif lang == "java":
        suffixes = [".java"]
    else:
        suffixes = [".py", ".java"]

    files: List[Path] = []
    for suffix in suffixes:
        files.extend(project_root.rglob(f"*{suffix}"))
    return sorted(p for p in files if p.is_file())


def _extract_relevant_blocks(task: TaskObject, repo_files: List[Path], file_texts: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Minimal relevant block set.
    Current strategy:
    - target file full text
    - files containing target name / qualname parts
    - keep this simple and stable
    """
    blocks: List[Dict[str, Any]] = []

    target_file = getattr(task, "target_file", None)
    target_name = getattr(task, "target_name", None)
    qualname = getattr(task, "qualname", None)

    if isinstance(target_file, str) and target_file in file_texts:
        blocks.append({
            "kind": "target_file",
            "file_path": target_file,
            "content": file_texts[target_file],
        })

    needles = [x for x in [target_name, qualname] if isinstance(x, str) and x.strip()]
    seen = {b["file_path"] for b in blocks}

    for path_str, text in file_texts.items():
        if path_str in seen:
            continue
        if any(n in text for n in needles):
            blocks.append({
                "kind": "related_file",
                "file_path": path_str,
                "content": text,
            })

    return blocks


def _construct_project_index(
    *,
    task: TaskObject,
    project_root: Path,
) -> ProjectIndex:
    """
    Build ProjectIndex in a schema-tolerant way.
    """
    repo_files = _iter_repo_files(project_root, getattr(task, "lang", None))
    file_texts: Dict[str, str] = {}

    for path in repo_files:
        rel = str(path.relative_to(project_root)).replace("\\", "/")
        file_texts[rel] = _read_text(path)

    relevant_blocks = _extract_relevant_blocks(task, repo_files, file_texts)

    candidate_kwargs: Dict[str, Any] = {
        "project_root": str(project_root),
        "project_name": getattr(task, "project", None),
        "language": getattr(task, "lang", None),
        "files": list(file_texts.keys()),
        "file_texts": file_texts,
        "relevant_blocks": relevant_blocks,
        "target_file": getattr(task, "target_file", None),
        "qualname": getattr(task, "qualname", None),
    }

    allowed = _dataclass_field_names(ProjectIndex)
    if allowed:
        candidate_kwargs = {k: v for k, v in candidate_kwargs.items() if k in allowed}
    return ProjectIndex(**candidate_kwargs)  # type: ignore[arg-type]


class LocalRepoTaskAdapter(TaskAdapter):
    """
    Build TaskObject + ProjectIndex from local repository files and raw task record.
    """

    def load_task(self, raw_task: Dict[str, Any]) -> TaskObject:
        return _construct_task_object(raw_task)

    def build_project_index(
        self,
        *,
        task: TaskObject,
        project_root: str | Path,
    ) -> ProjectIndex:
        root = Path(project_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"project_root not found: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"project_root is not a directory: {root}")
        return _construct_project_index(task=task, project_root=root)

    def load(
        self,
        *,
        raw_task: Dict[str, Any],
        project_root: str | Path,
    ) -> Tuple[TaskObject, ProjectIndex]:
        task = self.load_task(raw_task)
        project_index = self.build_project_index(task=task, project_root=project_root)
        return task, project_index