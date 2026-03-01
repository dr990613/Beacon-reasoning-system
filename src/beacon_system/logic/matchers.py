# src/beacon_system/logic/matchers.py
# -*- coding: utf-8 -*-
"""
matchers.py

Executable, testable, deterministic AST matchers for Constraints.match_spec.

Key rule:
- Verifier MUST NOT "rebuild reasoning". It only runs these matchers over candidate code.

MVP primitives:
- HasCall(func_name)
- HasImport(module) / HasName(name)
- ForbidCall(func_name) / ForbidPattern(regex)
- CallChain([a,b,c])  (MVP: within same function body, order-insensitive or order-sensitive flag)
- AssignFromCall(var, func)

Interface:
    match(code_ast: ast.AST, spec: MatchSpec) -> list[Violation]
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


# ----------------------------
# Violation model
# ----------------------------

class ViolationKind(str, Enum):
    MISSING = "missing"
    FORBIDDEN = "forbidden"
    MISMATCH = "mismatch"


@dataclass(frozen=True, slots=True)
class Violation:
    kind: ViolationKind
    spec_type: str
    message: str
    meta: Dict[str, Any] = field(default_factory=dict)


# ----------------------------
# MatchSpec primitives
# ----------------------------

@dataclass(frozen=True, slots=True)
class MatchSpec:
    """Base class for all match specs."""
    pass


@dataclass(frozen=True, slots=True)
class HasCall(MatchSpec):
    func_name: str


@dataclass(frozen=True, slots=True)
class HasImport(MatchSpec):
    module: str  # e.g. "math" or "numpy"


@dataclass(frozen=True, slots=True)
class HasName(MatchSpec):
    name: str


@dataclass(frozen=True, slots=True)
class ForbidCall(MatchSpec):
    func_name: str


@dataclass(frozen=True, slots=True)
class ForbidPattern(MatchSpec):
    regex: str
    flags: int = re.MULTILINE


@dataclass(frozen=True, slots=True)
class CallChain(MatchSpec):
    chain: Tuple[str, ...]
    order_sensitive: bool = False
    within_same_function: bool = True  # MVP: only supports True


@dataclass(frozen=True, slots=True)
class AssignFromCall(MatchSpec):
    var: str
    func_name: str


# ----------------------------
# AST extraction helpers
# ----------------------------

def _get_call_name(call: ast.Call) -> Optional[str]:
    """
    Best-effort call target name extraction:
    - foo(...) -> "foo"
    - obj.foo(...) -> "foo"
    """
    fn = call.func
    if isinstance(fn, ast.Name):
        return fn.id
    if isinstance(fn, ast.Attribute):
        return fn.attr
    return None


def _collect_calls(node: ast.AST) -> List[str]:
    """
    Collect call names in traversal order (ast.walk order is stable for a given AST).
    """
    out: List[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            name = _get_call_name(n)
            if name:
                out.append(name)
    return out


def _collect_imports(module: ast.AST) -> List[str]:
    """
    Collect imported top-level module names (best-effort).
    - import math -> "math"
    - import numpy as np -> "numpy"
    - from math import sqrt -> "math"
    """
    out: List[str] = []
    for n in ast.walk(module):
        if isinstance(n, ast.Import):
            for alias in n.names:
                if alias.name:
                    out.append(alias.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            if n.module:
                out.append(n.module.split(".")[0])
    return out


def _collect_names(node: ast.AST) -> List[str]:
    out: List[str] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.append(n.id)
    return out


def _collect_assign_from_call(node: ast.AST) -> List[Tuple[str, str]]:
    """
    Collect pairs (var, call_name) for simple assignment-from-call patterns.
    - x = foo(...)
    - x: T = foo(...)
    """
    pairs: List[Tuple[str, str]] = []
    for n in ast.walk(node):
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            call_name = _get_call_name(n.value)
            if not call_name:
                continue
            if len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
                pairs.append((n.targets[0].id, call_name))
        elif isinstance(n, ast.AnnAssign) and isinstance(n.value, ast.Call) and isinstance(n.target, ast.Name):
            call_name = _get_call_name(n.value)
            if call_name:
                pairs.append((n.target.id, call_name))
    return pairs


def _function_scopes(module_ast: ast.AST) -> List[ast.AST]:
    """
    Return function nodes (FunctionDef/AsyncFunctionDef) as separate scopes.
    If no function exists, treat whole module as a single scope.
    """
    fns: List[ast.AST] = []
    for n in ast.walk(module_ast):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fns.append(n)
    return fns or [module_ast]


# ----------------------------
# Matcher implementations
# ----------------------------

def _match_has_call(code_ast: ast.AST, spec: HasCall) -> List[Violation]:
    calls = _collect_calls(code_ast)
    if spec.func_name not in calls:
        return [Violation(
            kind=ViolationKind.MISSING,
            spec_type="HasCall",
            message=f"Missing call to '{spec.func_name}'.",
            meta={"func_name": spec.func_name},
        )]
    return []


def _match_has_import(code_ast: ast.AST, spec: HasImport) -> List[Violation]:
    imports = _collect_imports(code_ast)
    if spec.module not in imports:
        return [Violation(
            kind=ViolationKind.MISSING,
            spec_type="HasImport",
            message=f"Missing import of module '{spec.module}'.",
            meta={"module": spec.module},
        )]
    return []


def _match_has_name(code_ast: ast.AST, spec: HasName) -> List[Violation]:
    names = _collect_names(code_ast)
    if spec.name not in names:
        return [Violation(
            kind=ViolationKind.MISSING,
            spec_type="HasName",
            message=f"Missing name '{spec.name}'.",
            meta={"name": spec.name},
        )]
    return []


def _match_forbid_call(code_ast: ast.AST, spec: ForbidCall) -> List[Violation]:
    calls = _collect_calls(code_ast)
    if spec.func_name in calls:
        return [Violation(
            kind=ViolationKind.FORBIDDEN,
            spec_type="ForbidCall",
            message=f"Forbidden call to '{spec.func_name}' is present.",
            meta={"func_name": spec.func_name},
        )]
    return []


def _match_forbid_pattern(code_ast: ast.AST, spec: ForbidPattern) -> List[Violation]:
    # Deterministic: require source text in meta? Here we can only use ast.unparse if available (py3.9+).
    try:
        src = ast.unparse(code_ast)
    except Exception:
        # fallback: dump AST
        src = ast.dump(code_ast, include_attributes=False)

    if re.search(spec.regex, src, flags=spec.flags):
        return [Violation(
            kind=ViolationKind.FORBIDDEN,
            spec_type="ForbidPattern",
            message=f"Forbidden pattern /{spec.regex}/ matched.",
            meta={"regex": spec.regex},
        )]
    return []


def _match_call_chain(code_ast: ast.AST, spec: CallChain) -> List[Violation]:
    if not spec.chain:
        return []

    # MVP: within same function scope
    scopes = _function_scopes(code_ast) if spec.within_same_function else [code_ast]
    target = list(spec.chain)

    def contains_chain(calls: List[str]) -> bool:
        if spec.order_sensitive:
            # order-sensitive subsequence
            j = 0
            for c in calls:
                if c == target[j]:
                    j += 1
                    if j == len(target):
                        return True
            return False
        else:
            # order-insensitive: set containment
            return set(target).issubset(set(calls))

    for scope in scopes:
        calls = _collect_calls(scope)
        if contains_chain(calls):
            return []

    return [Violation(
        kind=ViolationKind.MISSING,
        spec_type="CallChain",
        message=f"Missing call chain {list(spec.chain)} (order_sensitive={spec.order_sensitive}).",
        meta={"chain": list(spec.chain), "order_sensitive": spec.order_sensitive},
    )]


def _match_assign_from_call(code_ast: ast.AST, spec: AssignFromCall) -> List[Violation]:
    pairs = _collect_assign_from_call(code_ast)
    if (spec.var, spec.func_name) not in pairs:
        return [Violation(
            kind=ViolationKind.MISSING,
            spec_type="AssignFromCall",
            message=f"Missing assignment '{spec.var} = {spec.func_name}(...)'.",
            meta={"var": spec.var, "func_name": spec.func_name, "observed": pairs},
        )]
    return []


# ----------------------------
# Public API
# ----------------------------

def match(code_ast: ast.AST, spec: MatchSpec) -> List[Violation]:
    """
    Execute a single MatchSpec against an AST and return violations.

    Determinism:
    - No random operations.
    - Traversal is based on ast.walk / deterministic scope extraction.
    """
    if isinstance(spec, HasCall):
        return _match_has_call(code_ast, spec)
    if isinstance(spec, HasImport):
        return _match_has_import(code_ast, spec)
    if isinstance(spec, HasName):
        return _match_has_name(code_ast, spec)
    if isinstance(spec, ForbidCall):
        return _match_forbid_call(code_ast, spec)
    if isinstance(spec, ForbidPattern):
        return _match_forbid_pattern(code_ast, spec)
    if isinstance(spec, CallChain):
        return _match_call_chain(code_ast, spec)
    if isinstance(spec, AssignFromCall):
        return _match_assign_from_call(code_ast, spec)

    return [Violation(
        kind=ViolationKind.MISMATCH,
        spec_type=type(spec).__name__,
        message=f"Unknown MatchSpec type: {type(spec).__name__}",
        meta={},
    )]


def match_all(code_ast: ast.AST, specs: Sequence[MatchSpec]) -> List[Violation]:
    """
    Convenience: execute multiple specs and concatenate violations deterministically.
    """
    violations: List[Violation] = []
    for s in specs:
        violations.extend(match(code_ast, s))
    return violations