# -*- coding: utf-8 -*-

"""
Rebuilder Agent

Responsibilities:
- Write generated code back into the full task/program context
- Build patched program
- Re-run logic engine on patched task
- Return rebuilt beacon and diagnostics

Stable external engine contract:
    build(task, project_index, run_config) -> LogicBuildResult

Design goals:
- No verification logic here
- No planning/scoring logic here
- Rebuild is explicit and debuggable
- Patch strategy is simple and stable
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from typing import Any, Dict, Optional, Tuple
import copy
import re


# ============================================================
# helpers
# ============================================================

def _as_dict(obj: Any) -> Dict[str, Any]:
    """
    Tolerant object-to-dict conversion.
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        try:
            return dict(vars(obj))
        except Exception:
            pass
    return {}


def _safe_int(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _get_attr_or_key(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _has_dataclass_instance(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__")


def _task_clone(task: Any) -> Any:
    """
    Deep copy task safely.
    """
    try:
        return copy.deepcopy(task)
    except Exception:
        return task


def _task_set(task: Any, key: str, value: Any) -> Any:
    """
    Set field on dict-like or object-like task.
    """
    if isinstance(task, dict):
        task[key] = value
        return task

    if _has_dataclass_instance(task):
        try:
            return replace(task, **{key: value})
        except Exception:
            pass

    try:
        setattr(task, key, value)
    except Exception:
        pass
    return task


def _normalize_newline(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _split_lines_keepends(text: str) -> list[str]:
    return _normalize_newline(text).splitlines(keepends=True)


def _ensure_trailing_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


# ============================================================
# patch strategies
# ============================================================

def _patch_by_lineno(file_content: str, generated_code: str, lineno: int, end_lineno: int) -> str:
    """
    Replace code region [lineno, end_lineno] in file_content using 1-based line numbers.
    """
    lines = _split_lines_keepends(file_content)
    start_idx = max(0, lineno - 1)
    end_idx = max(start_idx, end_lineno)

    new_block = _ensure_trailing_newline(_normalize_newline(generated_code))
    new_lines = _split_lines_keepends(new_block)

    patched = lines[:start_idx] + new_lines + lines[end_idx:]
    return "".join(patched)


def _extract_function_name(task: Any) -> Optional[str]:
    for key in ("target_function", "function_name", "entry_function", "name"):
        value = _get_attr_or_key(task, key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_signature(task: Any) -> Optional[str]:
    value = _get_attr_or_key(task, "signature")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _py_function_pattern(function_name: str) -> re.Pattern[str]:
    """
    Match a top-level Python function block conservatively.
    """
    pattern = (
        rf"(?ms)^def\s+{re.escape(function_name)}\s*\(.*?\)\s*:\s*\n"
        rf"(?:^[ \t]+.*\n|^\n)*"
    )
    return re.compile(pattern)


def _java_method_pattern(function_name: str) -> re.Pattern[str]:
    """
    Very simple Java method matcher.
    Works for many benchmark-style method bodies.
    """
    pattern = (
        rf"(?ms)"
        rf"^[ \t]*(?:public|private|protected)?[ \t]*(?:static[ \t]+)?"
        rf".*?\b{re.escape(function_name)}\s*\(.*?\)\s*\{{.*?^\}}"
    )
    return re.compile(pattern)


def _patch_by_function_name(file_content: str, generated_code: str, function_name: str) -> Tuple[str, bool]:
    """
    Fallback patch by function name for Python / Java-like code.
    """
    normalized = _normalize_newline(file_content)
    new_code = _ensure_trailing_newline(_normalize_newline(generated_code))

    py_pat = _py_function_pattern(function_name)
    if py_pat.search(normalized):
        return py_pat.sub(new_code, normalized, count=1), True

    java_pat = _java_method_pattern(function_name)
    if java_pat.search(normalized):
        return java_pat.sub(new_code, normalized, count=1), True

    return normalized, False


def _build_minimal_program(task: Any, generated_code: str) -> str:
    """
    Last-resort patched program if no full file content exists.
    """
    parts = []

    signature = _extract_signature(task)
    docstring = _get_attr_or_key(task, "docstring")
    original_context = _get_attr_or_key(task, "all_context") or _get_attr_or_key(task, "code_context")

    if isinstance(original_context, str) and original_context.strip():
        parts.append(str(original_context).strip())

    if isinstance(docstring, str) and docstring.strip():
        parts.append(f'"""\n{docstring.strip()}\n"""')

    if isinstance(signature, str) and signature.strip():
        parts.append(f"# signature: {signature.strip()}")

    parts.append(_normalize_newline(generated_code).strip())
    return "\n\n".join(p for p in parts if p).strip() + "\n"


def build_patched_program(task: Any, generated_code: str) -> Tuple[str, Dict[str, Any]]:
    """
    Build patched program from task + generated code.

    Patch order:
    1. file_content + lineno/end_lineno
    2. file_content + function_name
    3. original code replacement in file_content
    4. fallback minimal program
    """
    task_dict = _as_dict(task)

    file_content = task_dict.get("file_content")
    lineno = _safe_int(task_dict.get("lineno"))
    end_lineno = _safe_int(task_dict.get("end_lineno"))
    original_code = task_dict.get("code")
    function_name = _extract_function_name(task)

    diagnostics: Dict[str, Any] = {
        "patch_strategy": None,
        "used_file_content": False,
        "used_line_span": False,
        "used_function_name": False,
        "fallback_used": False,
        "patch_success": False,
    }

    if isinstance(file_content, str) and file_content.strip():
        diagnostics["used_file_content"] = True

        if lineno is not None and end_lineno is not None and lineno > 0 and end_lineno >= lineno:
            patched = _patch_by_lineno(file_content, generated_code, lineno, end_lineno)
            diagnostics["patch_strategy"] = "line_span_replace"
            diagnostics["used_line_span"] = True
            diagnostics["patch_success"] = True
            return patched, diagnostics

        if function_name:
            patched, ok = _patch_by_function_name(file_content, generated_code, function_name)
            if ok:
                diagnostics["patch_strategy"] = "function_name_replace"
                diagnostics["used_function_name"] = True
                diagnostics["patch_success"] = True
                return patched, diagnostics

        if isinstance(original_code, str) and original_code.strip() and original_code in file_content:
            patched = file_content.replace(original_code, _ensure_trailing_newline(generated_code), 1)
            diagnostics["patch_strategy"] = "exact_code_replace"
            diagnostics["patch_success"] = True
            return patched, diagnostics

    patched = _build_minimal_program(task, generated_code)
    diagnostics["patch_strategy"] = "minimal_fallback_program"
    diagnostics["fallback_used"] = True
    diagnostics["patch_success"] = True
    return patched, diagnostics


# ============================================================
# result objects
# ============================================================

@dataclass
class RebuildResult:
    patched_program: str
    rebuilt_beacon: Any
    rebuild_diagnostics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# agent
# ============================================================

class RebuilderAgent:
    """
    Bridge agent between generation and structure validation.

    Responsibilities:
    - patch generated code back into task context
    - rebuild logic beacon by calling engine.build(...)
    """

    def __init__(self, logic_engine: Any) -> None:
        """
        logic_engine must expose:
            build(task, project_index, run_config) -> LogicBuildResult
        """
        if not hasattr(logic_engine, "build"):
            raise TypeError("logic_engine must expose a stable build(task, project_index, run_config) method.")
        self.logic_engine = logic_engine

    def run(
        self,
        *,
        task: Any,
        generated_code: str,
        project_index: Any,
        run_config: Any,
    ) -> RebuildResult:
        """
        Patch generated code into the task and rebuild beacon.
        """
        if not isinstance(generated_code, str) or not generated_code.strip():
            raise ValueError("generated_code must be a non-empty string.")

        patched_program, patch_diag = build_patched_program(task, generated_code)

        patched_task = _task_clone(task)
        patched_task = _task_set(patched_task, "patched_program", patched_program)

        # Keep common compatibility fields for downstream logic
        if _get_attr_or_key(patched_task, "file_content", None) is not None:
            patched_task = _task_set(patched_task, "file_content", patched_program)

        if _get_attr_or_key(patched_task, "code", None) is not None:
            patched_task = _task_set(patched_task, "code", generated_code)

        rebuilt = self.logic_engine.build(
            patched_task,
            project_index,
            run_config,
        )

        rebuild_diagnostics = {
            "patch": patch_diag,
            "engine_called": True,
            "engine_contract": "build(task, project_index, run_config) -> LogicBuildResult",
            "generated_code_nonempty": bool(generated_code.strip()),
        }

        return RebuildResult(
            patched_program=patched_program,
            rebuilt_beacon=rebuilt,
            rebuild_diagnostics=rebuild_diagnostics,
        )