# src/beacon_system/logic/anchors.py
# -*- coding: utf-8 -*-
"""
anchors.py

Anchor/NodeID/Namespace are the identity layer for Beacon Logic.

Design goals (MVP):
- Stable, serializable, comparable identities for AST nodes and symbols.
- "Enough information" in Anchor to uniquely locate code (file + qualname + position + namespace).
- NodeID is a stable string derived from anchor + ast_kind + local_index.
  (Deterministic formatting; hashing is optional but recommended for compactness.)

IMPORTANT:
- Determinism is ultimately enforced by normalize.py; anchors.py only provides stable primitives.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, NewType, Dict, Any


NodeID = NewType("NodeID", str)


class Namespace(str, Enum):
    """Where an anchor lives (scope)."""

    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    LOCAL = "local"
    GLOBAL = "global"


@dataclass(frozen=True, slots=True)
class Anchor:
    """
    A stable location descriptor for a source element.

    Fields:
    - file: normalized file path (as provided by caller; normalize.py may canonicalize later)
    - qualname: qualified name, e.g. "pkg.mod:Class.method" or "mod.func"
    - lineno/col: 1-based line number and 0-based column offset when available
    - end_lineno/end_col: optional end position (Python 3.8+ nodes can have these)
    - namespace: coarse scope label
    """

    file: str
    qualname: str
    lineno: int
    col: int
    end_lineno: Optional[int] = None
    end_col: Optional[int] = None
    namespace: Namespace = Namespace.LOCAL

    def span(self) -> Tuple[int, int, Optional[int], Optional[int]]:
        return (self.lineno, self.col, self.end_lineno, self.end_col)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file": self.file,
            "qualname": self.qualname,
            "lineno": self.lineno,
            "col": self.col,
            "end_lineno": self.end_lineno,
            "end_col": self.end_col,
            "namespace": self.namespace.value,
        }


def _infer_namespace_from_qualname(qualname: str) -> Namespace:
    """
    Heuristic inference:
    - empty qualname => MODULE
    - contains "." (or ":") usually => FUNCTION/CLASS (ambiguous)
    We keep it conservative and default to FUNCTION if any delimiter exists.
    Caller may override with an explicit namespace in future.
    """
    q = (qualname or "").strip()
    if not q:
        return Namespace.MODULE
    # Very light heuristic: if qualname contains "Class." pattern we still label FUNCTION for now.
    if "." in q or ":" in q:
        return Namespace.FUNCTION
    return Namespace.MODULE


def anchor_of(node: ast.AST, file: str, qualname: str) -> Anchor:
    """
    Create an Anchor for an AST node.

    Requirements:
    - Works even if node has no position (lineno/col_offset). In that case uses (0, 0).
    - Includes end positions when present (end_lineno/end_col_offset).
    """
    lineno = int(getattr(node, "lineno", 0) or 0)
    col = int(getattr(node, "col_offset", 0) or 0)

    end_lineno = getattr(node, "end_lineno", None)
    end_col = getattr(node, "end_col_offset", None)

    # Normalize missing end positions to None (not 0), to reduce ambiguity.
    if end_lineno is not None:
        end_lineno = int(end_lineno)
    if end_col is not None:
        end_col = int(end_col)

    ns = _infer_namespace_from_qualname(qualname)
    return Anchor(
        file=file,
        qualname=qualname,
        lineno=lineno,
        col=col,
        end_lineno=end_lineno,
        end_col=end_col,
        namespace=ns,
    )


def make_node_id(anchor: Anchor, ast_kind: str, local_index: int) -> NodeID:
    """
    Deterministically build a NodeID string from:
      file + qualname + lineno + col + end_lineno + end_col + namespace + ast_kind + local_index

    NodeID format:
      "bcn:<sha1>:<lineno>:<col>:<ast_kind>:<local_index>"
    where <sha1> hashes the stable prefix to keep IDs compact.

    NOTE:
    - local_index is used to disambiguate multiple nodes sharing the same position.
    - normalize.py may further canonicalize or remap NodeIDs; do NOT change this format lightly.
    """
    if local_index < 0:
        raise ValueError("local_index must be non-negative")

    kind = (ast_kind or "").strip() or "AST"
    prefix = (
        f"{anchor.file}|{anchor.qualname}|{anchor.lineno}|{anchor.col}|"
        f"{anchor.end_lineno}|{anchor.end_col}|{anchor.namespace.value}|{kind}|{local_index}"
    )
    digest = hashlib.sha1(prefix.encode("utf-8")).hexdigest()  # stable, compact
    node_id = f"bcn:{digest}:{anchor.lineno}:{anchor.col}:{kind}:{local_index}"
    return NodeID(node_id)


def ast_kind(node: ast.AST) -> str:
    """Return a stable AST kind label (class name)."""
    return type(node).__name__


def stable_anchor_key(anchor: Anchor) -> Tuple:
    """
    A comparable/sortable key for anchors. Useful in normalize.py.
    """
    return (
        anchor.file,
        anchor.qualname,
        anchor.lineno,
        anchor.col,
        anchor.end_lineno if anchor.end_lineno is not None else -1,
        anchor.end_col if anchor.end_col is not None else -1,
        anchor.namespace.value,
    )