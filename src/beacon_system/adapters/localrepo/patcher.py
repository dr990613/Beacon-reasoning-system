# src/beacon_system/adapters/localrepo/patcher.py
# -*- coding: utf-8 -*-

"""
LocalRepo patcher (minimal, replay-friendly)

- Only responsibility: locate target qualname (function / method) in a Python file and replace its body.
- We do NOT attempt perfect AST rewrite; we aim for robust minimal replacement.

Supported qualname forms:
- "foo"                 (top-level function)
- "ClassName.method"    (method inside class)
- "A.B.method"          (nested classes supported if present)

Replacement strategy:
- Parse AST to find the node (FunctionDef/AsyncFunctionDef) matching qualname.
- Compute source line span from node.lineno..node.end_lineno (Python 3.8+)
- Replace that span with `new_code` (assumed to contain a full def ... block).
"""

from __future__ import annotations

import ast
import os
from typing import List, Tuple


class PatchError(RuntimeError):
    pass


def _read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().splitlines(True)  # keep \n


def _write_lines(path: str, lines: List[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _qual_parts(qualname: str) -> List[str]:
    return [p for p in (qualname or "").split(".") if p.strip()]


def _find_function_span(tree: ast.AST, qualname: str) -> Tuple[int, int]:
    """
    Return (start_line_idx, end_line_idx_exclusive) 0-based indices for replacement.
    """
    parts = _qual_parts(qualname)
    if not parts:
        raise PatchError("Empty qualname")

    # We walk with a class stack to build qualified names.
    spans: List[Tuple[str, int, int]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: List[str] = []

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            q = ".".join(self.stack + [node.name]) if self.stack else node.name
            if hasattr(node, "lineno") and hasattr(node, "end_lineno") and node.end_lineno is not None:
                spans.append((q, node.lineno, node.end_lineno))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            q = ".".join(self.stack + [node.name]) if self.stack else node.name
            if hasattr(node, "lineno") and hasattr(node, "end_lineno") and node.end_lineno is not None:
                spans.append((q, node.lineno, node.end_lineno))
            self.generic_visit(node)

    Visitor().visit(tree)

    target = ".".join(parts)
    for q, lineno, end_lineno in spans:
        if q == target:
            # Convert 1-based lineno to 0-based slice
            return lineno - 1, end_lineno
    raise PatchError(f"Target qualname not found: {qualname}")


def apply_patch(repo_dir: str, target_file: str, target_qualname: str, new_code: str) -> None:
    repo_dir = os.path.abspath(repo_dir)
    file_path = os.path.join(repo_dir, target_file)
    if not os.path.isfile(file_path):
        raise PatchError(f"Target file not found: {file_path}")

    src = "".join(_read_lines(file_path))
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        raise PatchError(f"Cannot parse target file: {e.msg} at line {e.lineno}") from e

    start, end = _find_function_span(tree, target_qualname)

    lines = src.splitlines(True)  # keep \n
    new_block = (new_code or "").rstrip() + "\n"
    new_lines = new_block.splitlines(True)

    # Replace the function/method block
    patched = lines[:start] + new_lines + lines[end:]
    _write_lines(file_path, patched)