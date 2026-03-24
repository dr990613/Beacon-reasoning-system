# -*- coding: utf-8 -*-

"""
Stable local patch infrastructure.

Responsibilities:
- Inject generated code into target file/content
- Keep patch behavior stable, replayable, and revertible
- Provide infrastructure for Rebuilder / runtime prep
- No Beacon logic here
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import hashlib
import shutil
import re


@dataclass(frozen=True)
class PatchPlan:
    """
    Replayable patch description.
    """
    file_path: str
    strategy: str
    lineno: Optional[int] = None
    end_lineno: Optional[int] = None
    target_name: Optional[str] = None
    language: Optional[str] = None


@dataclass(frozen=True)
class PatchResult:
    ok: bool
    patched_text: str
    original_text: str
    plan: PatchPlan
    diagnostics: Dict[str, Any]


@dataclass(frozen=True)
class AppliedPatch:
    ok: bool
    target_path: str
    backup_path: Optional[str]
    result: PatchResult


def _normalize_newline(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _safe_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _patch_by_line_span(
    *,
    original_text: str,
    generated_code: str,
    lineno: int,
    end_lineno: int,
) -> str:
    lines = _normalize_newline(original_text).splitlines(keepends=True)
    start_idx = max(0, lineno - 1)
    end_idx = max(start_idx, end_lineno)
    new_block = _ensure_trailing_newline(_normalize_newline(generated_code))
    new_lines = new_block.splitlines(keepends=True)
    return "".join(lines[:start_idx] + new_lines + lines[end_idx:])


def _py_function_pattern(function_name: str) -> re.Pattern[str]:
    pattern = (
        rf"(?ms)^def\s+{re.escape(function_name)}\s*\(.*?\)\s*:\s*\n"
        rf"(?:^[ \t]+.*\n|^\n)*"
    )
    return re.compile(pattern)


def _java_method_pattern(function_name: str) -> re.Pattern[str]:
    pattern = (
        rf"(?ms)"
        rf"^[ \t]*(?:public|private|protected)?[ \t]*(?:static[ \t]+)?"
        rf".*?\b{re.escape(function_name)}\s*\(.*?\)\s*\{{.*?^\}}"
    )
    return re.compile(pattern)


def _patch_by_target_name(
    *,
    original_text: str,
    generated_code: str,
    target_name: str,
    language: Optional[str],
) -> Tuple[str, bool]:
    normalized = _normalize_newline(original_text)
    new_code = _ensure_trailing_newline(_normalize_newline(generated_code))

    patterns = []
    if language == "python":
        patterns = [_py_function_pattern(target_name)]
    elif language == "java":
        patterns = [_java_method_pattern(target_name)]
    else:
        patterns = [_py_function_pattern(target_name), _java_method_pattern(target_name)]

    for pat in patterns:
        if pat.search(normalized):
            return pat.sub(new_code, normalized, count=1), True

    return normalized, False


def build_patch_result(
    *,
    original_text: str,
    generated_code: str,
    file_path: str,
    language: Optional[str] = None,
    lineno: Optional[int] = None,
    end_lineno: Optional[int] = None,
    target_name: Optional[str] = None,
) -> PatchResult:
    """
    Pure patch builder.
    Does not write to disk.
    """
    original_text = _normalize_newline(original_text)
    generated_code = _normalize_newline(generated_code)

    if lineno is not None and end_lineno is not None and lineno > 0 and end_lineno >= lineno:
        patched = _patch_by_line_span(
            original_text=original_text,
            generated_code=generated_code,
            lineno=lineno,
            end_lineno=end_lineno,
        )
        plan = PatchPlan(
            file_path=file_path,
            strategy="line_span_replace",
            lineno=lineno,
            end_lineno=end_lineno,
            target_name=target_name,
            language=language,
        )
        diagnostics = {
            "patch_success": True,
            "strategy": "line_span_replace",
            "original_sha256": _text_sha256(original_text),
            "patched_sha256": _text_sha256(patched),
        }
        return PatchResult(
            ok=True,
            patched_text=patched,
            original_text=original_text,
            plan=plan,
            diagnostics=diagnostics,
        )

    if isinstance(target_name, str) and target_name.strip():
        patched, ok = _patch_by_target_name(
            original_text=original_text,
            generated_code=generated_code,
            target_name=target_name.strip(),
            language=language,
        )
        plan = PatchPlan(
            file_path=file_path,
            strategy="target_name_replace",
            lineno=lineno,
            end_lineno=end_lineno,
            target_name=target_name,
            language=language,
        )
        diagnostics = {
            "patch_success": bool(ok),
            "strategy": "target_name_replace",
            "original_sha256": _text_sha256(original_text),
            "patched_sha256": _text_sha256(patched),
        }
        return PatchResult(
            ok=bool(ok),
            patched_text=patched,
            original_text=original_text,
            plan=plan,
            diagnostics=diagnostics,
        )

    plan = PatchPlan(
        file_path=file_path,
        strategy="no_patch",
        lineno=lineno,
        end_lineno=end_lineno,
        target_name=target_name,
        language=language,
    )
    diagnostics = {
        "patch_success": False,
        "strategy": "no_patch",
        "reason": "insufficient patch coordinates",
        "original_sha256": _text_sha256(original_text),
        "patched_sha256": _text_sha256(original_text),
    }
    return PatchResult(
        ok=False,
        patched_text=original_text,
        original_text=original_text,
        plan=plan,
        diagnostics=diagnostics,
    )


def apply_patch_to_file(
    *,
    target_path: str | Path,
    patch_result: PatchResult,
    make_backup: bool = True,
) -> AppliedPatch:
    """
    Write patched text to target file and optionally create backup.
    """
    path = Path(target_path)
    if not path.exists():
        raise FileNotFoundError(f"target_path not found: {path}")

    backup_path: Optional[Path] = None
    if make_backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup_path)

    path.write_text(patch_result.patched_text, encoding="utf-8")

    return AppliedPatch(
        ok=True,
        target_path=str(path),
        backup_path=None if backup_path is None else str(backup_path),
        result=patch_result,
    )


def rollback_patch(
    *,
    target_path: str | Path,
    backup_path: str | Path,
) -> None:
    """
    Restore file from backup.
    """
    target = Path(target_path)
    backup = Path(backup_path)
    if not backup.exists():
        raise FileNotFoundError(f"backup_path not found: {backup}")
    shutil.copy2(backup, target)


def patch_file_from_task(
    *,
    task: Any,
    generated_code: str,
    repo_root: str | Path,
    make_backup: bool = True,
) -> AppliedPatch:
    """
    Convenience patch entry using common TaskObject fields.

    Expected task fields:
    - target_file or file_path
    - lang
    - target_name / target_function / function_name / name
    - lineno / end_lineno
    """
    file_path = (
        getattr(task, "target_file", None)
        or getattr(task, "file_path", None)
    )
    if not isinstance(file_path, str) or not file_path.strip():
        raise ValueError("task must contain target_file or file_path for patching.")

    target_path = Path(repo_root) / file_path
    original_text = target_path.read_text(encoding="utf-8")

    target_name = (
        getattr(task, "target_name", None)
        or getattr(task, "target_function", None)
        or getattr(task, "function_name", None)
        or getattr(task, "name", None)
    )

    patch_result = build_patch_result(
        original_text=original_text,
        generated_code=generated_code,
        file_path=file_path,
        language=getattr(task, "lang", None),
        lineno=_safe_int(getattr(task, "lineno", None)),
        end_lineno=_safe_int(getattr(task, "end_lineno", None)),
        target_name=target_name,
    )
    if not patch_result.ok:
        raise RuntimeError(f"Patch failed: {patch_result.diagnostics}")

    return apply_patch_to_file(
        target_path=target_path,
        patch_result=patch_result,
        make_backup=make_backup,
    )