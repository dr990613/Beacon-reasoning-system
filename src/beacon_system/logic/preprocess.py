# src/beacon_system/logic/preprocess.py
# -*- coding: utf-8 -*-

"""
Preprocess layer for Beacon Logic.

Design goals:
- ONLY do input cleaning / normalization / tolerance handling
- NO semantic reasoning, NO beacon inference, NO dependency analysis
- Support both Python and Java task inputs
- Produce a stable, simple intermediate structure for downstream logic rules

Typical usage:
    cleaned = preprocess_task(task)

Expected downstream consumers:
    - logic.rules_local
    - logic.rules_global
    - logic.engine

Notes:
- This module is intentionally schema-tolerant.
- It accepts dict-like task objects or dataclass-like objects.
- It does not assume the final `types.py` is already fixed.
- When `types.py` is ready, you can replace the result dict with a typed contract.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import ast
import codecs
import re
import textwrap
from typing import Any, Dict, List, Optional, Tuple


# ============================================================
# Simple local contracts
# Replace with types.py contracts later if needed
# ============================================================

@dataclass
class PreprocessResult:
    lang: str
    raw_text: str
    normalized_text: str
    class_level_text: str
    file_content_text: str
    import_lines: List[str]
    signature_lines: List[str]
    body_text: str
    target_name: Optional[str]
    target_signature: Optional[str]
    metadata: Dict[str, Any]
    warnings: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Public API
# ============================================================

def preprocess_task(task: Any) -> Dict[str, Any]:
    """
    Normalize task input into a stable structure.

    This function is intentionally tolerant:
    - task can be dict-like or object-like
    - supports Python and Java
    - supports escaped string payloads
    - supports partial task records

    Returns:
        dict with normalized fields for downstream logic stages
    """
    warnings: List[str] = []

    raw_lang = _read_field(task, ["lang", "language"])
    class_level = _read_field(task, ["class_level", "all_context", "context", "class_context"], default="")
    file_content = _read_field(task, ["file_content", "file", "source", "code_context"], default="")
    code = _read_field(task, ["code", "snippet", "implementation"], default="")
    name = _read_field(task, ["name", "target_name", "function_name", "method_name"], default=None)
    signature = _read_field(task, ["signature", "method_signature", "function_signature"], default=None)
    docstring = _read_field(task, ["docstring", "comment", "spec"], default="")
    file_path = _read_field(task, ["file_path", "path", "target_file"], default=None)
    project = _read_field(task, ["project", "repo", "repository"], default=None)

    class_level = _coerce_text(class_level)
    file_content = _coerce_text(file_content)
    code = _coerce_text(code)
    docstring = _coerce_text(docstring)
    signature = _coerce_optional_text(signature)
    name = _coerce_optional_text(name)

    # Decode escaped payloads like:
    # \"class_level\" : \"import ...\\n...\"
    class_level = _safe_unescape(class_level)
    file_content = _safe_unescape(file_content)
    code = _safe_unescape(code)
    docstring = _safe_unescape(docstring)
    if signature is not None:
        signature = _safe_unescape(signature)
    if name is not None:
        name = _safe_unescape(name)

    # Normalize line endings and indentation
    class_level = _normalize_text(class_level)
    file_content = _normalize_text(file_content)
    code = _normalize_text(code)
    docstring = _normalize_text(docstring)
    if signature is not None:
        signature = _normalize_text(signature)
    if name is not None:
        name = name.strip()

    # If file_content is empty, fall back to code
    merged_source = file_content if file_content.strip() else code
    merged_source = _normalize_text(merged_source)

    # Infer language
    lang = _infer_language(raw_lang=raw_lang, source=merged_source, class_level=class_level, signature=signature)
    if raw_lang is None:
        warnings.append(f"language inferred as '{lang}'")

    # Language-specific light normalization
    if lang == "python":
        class_level, merged_source, signature_lines, import_lines, target_name, target_signature, body_text, extra_warnings = (
            _preprocess_python(
                class_level=class_level,
                source=merged_source,
                fallback_name=name,
                fallback_signature=signature,
            )
        )
    elif lang == "java":
        class_level, merged_source, signature_lines, import_lines, target_name, target_signature, body_text, extra_warnings = (
            _preprocess_java(
                class_level=class_level,
                source=merged_source,
                fallback_name=name,
                fallback_signature=signature,
            )
        )
    else:
        # Keep tolerant fallback
        signature_lines = _extract_signature_like_lines(class_level + "\n" + merged_source)
        import_lines = _extract_import_like_lines(class_level + "\n" + merged_source)
        target_name = name
        target_signature = signature
        body_text = merged_source
        extra_warnings = [f"unsupported language '{lang}', used generic preprocessing"]
        lang = "unknown"

    warnings.extend(extra_warnings)

    result = PreprocessResult(
        lang=lang,
        raw_text=_join_non_empty([class_level, merged_source]),
        normalized_text=_join_non_empty([class_level, merged_source, docstring]).strip(),
        class_level_text=class_level,
        file_content_text=merged_source,
        import_lines=import_lines,
        signature_lines=signature_lines,
        body_text=body_text,
        target_name=target_name,
        target_signature=target_signature,
        metadata={
            "file_path": file_path,
            "project": project,
            "docstring": docstring,
            "has_class_level": bool(class_level.strip()),
            "has_file_content": bool(merged_source.strip()),
            "input_name": name,
            "input_signature": signature,
        },
        warnings=warnings,
    )
    return result.to_dict()


# ============================================================
# Python preprocessing
# ============================================================

def _preprocess_python(
    class_level: str,
    source: str,
    fallback_name: Optional[str],
    fallback_signature: Optional[str],
) -> Tuple[str, str, List[str], List[str], Optional[str], Optional[str], str, List[str]]:
    warnings: List[str] = []

    class_level = _normalize_python_text(class_level)
    source = _normalize_python_text(source)

    combined = _join_non_empty([class_level, source])

    import_lines = _extract_python_imports(combined)
    signature_lines = _extract_python_signatures(combined)

    target_name = fallback_name or _guess_python_target_name(signature_lines)
    target_signature = fallback_signature or _guess_signature_by_name(signature_lines, target_name)

    body_text = source if source.strip() else combined

    # Try AST parse for light validation only
    if body_text.strip():
        try:
            ast.parse(body_text)
        except SyntaxError:
            warnings.append("python source is not fully parseable after normalization; kept tolerant raw body")

    return (
        class_level,
        source,
        signature_lines,
        import_lines,
        target_name,
        target_signature,
        body_text,
        warnings,
    )


def _normalize_python_text(text: str) -> str:
    text = text.replace("\t", "    ")
    text = textwrap.dedent(text)
    text = _strip_leading_noise(text)
    return text.strip("\n")


def _extract_python_imports(text: str) -> List[str]:
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            lines.append(s)
    return _unique_keep_order(lines)


def _extract_python_signatures(text: str) -> List[str]:
    pattern = re.compile(r"^\s*(async\s+def|def|class)\s+[A-Za-z_][A-Za-z0-9_]*.*:\s*$", re.MULTILINE)
    return _unique_keep_order([m.group(0).strip() for m in pattern.finditer(text)])


def _guess_python_target_name(signature_lines: List[str]) -> Optional[str]:
    for line in signature_lines:
        m = re.match(r"^(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)", line)
        if m:
            return m.group(1)
    return None


# ============================================================
# Java preprocessing
# ============================================================

def _preprocess_java(
    class_level: str,
    source: str,
    fallback_name: Optional[str],
    fallback_signature: Optional[str],
) -> Tuple[str, str, List[str], List[str], Optional[str], Optional[str], str, List[str]]:
    warnings: List[str] = []

    class_level = _normalize_java_text(class_level)
    source = _normalize_java_text(source)

    combined = _join_non_empty([class_level, source])

    import_lines = _extract_java_imports(combined)
    signature_lines = _extract_java_signatures(class_level, source)

    target_name = fallback_name or _guess_java_target_name(signature_lines)
    target_signature = fallback_signature or _guess_signature_by_name(signature_lines, target_name)

    # For Java, prefer method/class body source if present; otherwise keep class-level summary text
    body_text = source if source.strip() else combined

    if not signature_lines and class_level.strip():
        warnings.append("java class-level context found but no signature-like line extracted")

    return (
        class_level,
        source,
        signature_lines,
        import_lines,
        target_name,
        target_signature,
        body_text,
        warnings,
    )


def _normalize_java_text(text: str) -> str:
    text = textwrap.dedent(text)
    text = _strip_leading_noise(text)
    return text.strip("\n")


def _extract_java_imports(text: str) -> List[str]:
    lines = []
    for line in text.splitlines():
        s = line.strip().rstrip(";")
        if s.startswith("import "):
            lines.append(s + ";")
    return _unique_keep_order(lines)


def _extract_java_signatures(class_level: str, source: str) -> List[str]:
    lines: List[str] = []

    # Class-level compact declarations like:
    # hasLength(String str);
    # Strings();
    # Charset UTF_8;
    for line in class_level.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("import "):
            continue
        if _looks_like_java_signature_or_member(s):
            lines.append(s if s.endswith(";") else s + ";")

    # Source-level method/class signatures
    source_pattern = re.compile(
        r"""
        ^\s*
        (?:
            (?:public|protected|private|static|final|native|synchronized|abstract|default|strictfp)\s+
        )*
        (?:
            class|interface|enum|
            [A-Za-z_<>\[\],\s?]+\s+[A-Za-z_][A-Za-z0-9_]*\s*
        )
        \([^)]*\)?\s*
        (?:throws\s+[^{]+)?\{
        |
        ^\s*
        (?:
            (?:public|protected|private|static|final|abstract)\s+
        )*
        (?:class|interface|enum)\s+[A-Za-z_][A-Za-z0-9_]*[^{]*\{
        """,
        re.MULTILINE | re.VERBOSE,
    )
    for m in source_pattern.finditer(source):
        sig = m.group(0).strip()
        lines.append(sig)

    return _unique_keep_order(lines)


def _looks_like_java_signature_or_member(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    if line.startswith(("class ", "interface ", "enum ")):
        return True
    if "(" in line and ")" in line:
        return True
    if re.match(r"^[A-Za-z_<>\[\]]+\s+[A-Za-z_][A-Za-z0-9_]*;?$", line):
        return True
    return False


def _guess_java_target_name(signature_lines: List[str]) -> Optional[str]:
    for line in signature_lines:
        # method or constructor
        m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*\(", line)
        if m:
            return m.group(1)
        # field
        m = re.search(r"[A-Za-z_<>\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*;?$", line)
        if m:
            return m.group(1)
    return None


# ============================================================
# Shared helpers
# ============================================================

def _read_field(obj: Any, names: List[str], default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_optional_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    value = str(value)
    return value if value.strip() else None


def _safe_unescape(text: str) -> str:
    """
    Tolerantly decode escaped content like:
        \\n, \\t, \\"
    but avoid breaking normal source whenever possible.
    """
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Only decode if escaped patterns are clearly present
    if "\\n" in text or '\\"' in text or "\\t" in text:
        try:
            return codecs.decode(text, "unicode_escape")
        except Exception:
            return text
    return text


def _normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.strip("\x00")
    return text


def _strip_leading_noise(text: str) -> str:
    """
    Remove obvious leading noise without doing semantic edits.
    Example:
        n\\n    def foo(...):
    """
    lines = text.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)

    if lines and lines[0].strip() in {"n", '"', "'"}:
        lines.pop(0)

    return "\n".join(lines)


def _infer_language(
    raw_lang: Optional[str],
    source: str,
    class_level: str,
    signature: Optional[str],
) -> str:
    if raw_lang:
        value = str(raw_lang).strip().lower()
        if value in {"python", "py"}:
            return "python"
        if value in {"java"}:
            return "java"

    joined = "\n".join([class_level, source, signature or ""]).strip()

    python_score = 0
    java_score = 0

    if re.search(r"^\s*def\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", joined, re.MULTILINE):
        python_score += 4
    if re.search(r"^\s*class\s+[A-Za-z_][A-Za-z0-9_]*\s*[:(]", joined, re.MULTILINE):
        python_score += 2
    if "self" in joined or "__init__" in joined or "elif " in joined:
        python_score += 2
    if re.search(r"^\s*from\s+\S+\s+import\s+", joined, re.MULTILINE):
        python_score += 2

    if "import java." in joined or "public class " in joined:
        java_score += 4
    if re.search(r"\b(public|private|protected|static|final)\b", joined):
        java_score += 2
    if re.search(r"[A-Za-z_][A-Za-z0-9_]*\s*\([^)]*\)\s*;", class_level):
        java_score += 2
    if re.search(r"\bthrows\b|\bimplements\b|\bextends\b", joined):
        java_score += 2

    if python_score >= java_score and python_score > 0:
        return "python"
    if java_score > python_score and java_score > 0:
        return "java"
    return "unknown"


def _extract_import_like_lines(text: str) -> List[str]:
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            lines.append(s)
    return _unique_keep_order(lines)


def _extract_signature_like_lines(text: str) -> List[str]:
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if "(" in s and ")" in s:
            lines.append(s)
    return _unique_keep_order(lines)


def _guess_signature_by_name(signature_lines: List[str], target_name: Optional[str]) -> Optional[str]:
    if not target_name:
        return signature_lines[0] if signature_lines else None
    for line in signature_lines:
        if re.search(rf"\b{re.escape(target_name)}\b", line):
            return line
    return signature_lines[0] if signature_lines else None


def _join_non_empty(parts: List[str]) -> str:
    return "\n".join([p for p in parts if p and p.strip()])


def _unique_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out