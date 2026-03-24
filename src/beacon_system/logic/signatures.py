# src/beacon_system/logic/signatures.py
# -*- coding: utf-8 -*-

"""
Extract function signatures, call argument hints, variable origins, and binding clues.

Design goals:
- support Python and Java
- give generator usable argument-binding hints
- avoid overly heavy schema
- focus on:
    - function signature
    - called API shape
    - key variable origins
    - parameter binding clues
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional


@dataclass
class SignatureHints:
    lang: str
    functions: List[Dict[str, Any]]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_signature_hints(preprocessed: Any, raw_ir: Any) -> Dict[str, Any]:
    pre = _as_dict(preprocessed)
    ir = _as_dict(raw_ir)

    lang = str(pre.get("lang", "unknown")).lower().strip()
    source = str(pre.get("body_text") or pre.get("file_content_text") or "")
    functions = ir.get("functions", []) or []

    if lang == "python":
        result = _build_python_signature_hints(source, functions)
    elif lang == "java":
        result = _build_java_signature_hints(source, functions)
    else:
        result = SignatureHints(lang=lang, functions=[], debug={"reason": "unsupported"}).to_dict()
    return result


def _build_python_signature_hints(source: str, functions: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        tree = ast.parse(source)
    except Exception:
        return SignatureHints(lang="python", functions=[], debug={"parse": "failed"}).to_dict()

    fn_nodes = {
        n.name: n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    out = []
    for fn in functions:
        fn_name = str(fn.get("function_name", ""))
        node = fn_nodes.get(fn_name)
        if node is None:
            continue

        signature = _python_signature(node)
        parameters = _python_parameters(node)
        call_hints = _python_call_hints(node)
        variable_origins = _python_variable_origins(node)
        return_hints = _python_return_hints(node)

        out.append(
            {
                "function_name": fn_name,
                "signature": signature,
                "parameters": parameters,
                "call_hints": call_hints,
                "variable_origins": variable_origins,
                "return_hints": return_hints,
            }
        )

    return SignatureHints(
        lang="python",
        functions=out,
        debug={"function_count": len(out)},
    ).to_dict()


def _build_java_signature_hints(source: str, functions: List[Dict[str, Any]]) -> Dict[str, Any]:
    methods = _extract_java_method_blocks(source)

    out = []
    fn_names = {str(f.get("function_name", "")) for f in functions}

    for method_name, signature, start_line, block_lines in methods:
        if method_name not in fn_names:
            continue

        parameters = _java_parameters(signature)
        call_hints = _java_call_hints(block_lines, start_line)
        variable_origins = _java_variable_origins(block_lines, start_line)
        return_hints = _java_return_hints(block_lines, start_line)

        out.append(
            {
                "function_name": method_name,
                "signature": signature,
                "parameters": parameters,
                "call_hints": call_hints,
                "variable_origins": variable_origins,
                "return_hints": return_hints,
            }
        )

    return SignatureHints(
        lang="java",
        functions=out,
        debug={"function_count": len(out)},
    ).to_dict()


# ---------------- Python ----------------

def _python_signature(fn: ast.AST) -> str:
    if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        parts = []
        for a in fn.args.args:
            parts.append(a.arg)
        if fn.args.vararg:
            parts.append("*" + fn.args.vararg.arg)
        for a in fn.args.kwonlyargs:
            parts.append(a.arg)
        if fn.args.kwarg:
            parts.append("**" + fn.args.kwarg.arg)
        prefix = "async def" if isinstance(fn, ast.AsyncFunctionDef) else "def"
        return f"{prefix} {fn.name}({', '.join(parts)})"
    return getattr(fn, "name", "<function>")


def _python_parameters(fn: ast.AST) -> List[Dict[str, Any]]:
    out = []
    if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return out
    for a in fn.args.args:
        out.append({"name": a.arg, "kind": "positional_or_keyword"})
    for a in fn.args.kwonlyargs:
        out.append({"name": a.arg, "kind": "kwonly"})
    if fn.args.vararg:
        out.append({"name": fn.args.vararg.arg, "kind": "vararg"})
    if fn.args.kwarg:
        out.append({"name": fn.args.kwarg.arg, "kind": "kwarg"})
    return out


def _python_call_hints(fn: ast.AST) -> List[Dict[str, Any]]:
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            callee = _python_call_name(node)
            args = []
            for a in node.args:
                args.append(_python_expr_text(a))
            for kw in node.keywords:
                if kw.arg is None:
                    args.append("**" + _python_expr_text(kw.value))
                else:
                    args.append(f"{kw.arg}={_python_expr_text(kw.value)}")
            out.append(
                {
                    "line_no": getattr(node, "lineno", -1),
                    "callee": callee,
                    "arguments": args,
                }
            )
    return out


def _python_variable_origins(fn: ast.AST) -> List[Dict[str, Any]]:
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            value_text = _python_expr_text(node.value)
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out.append(
                        {
                            "variable": t.id,
                            "line_no": getattr(node, "lineno", -1),
                            "origin": value_text,
                        }
                    )
    return out


def _python_return_hints(fn: ast.AST) -> List[Dict[str, Any]]:
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Return):
            out.append(
                {
                    "line_no": getattr(node, "lineno", -1),
                    "value": _python_expr_text(node.value) if node.value is not None else None,
                }
            )
    return out


def _python_expr_text(node: Optional[ast.AST]) -> str:
    if node is None:
        return "None"
    try:
        return ast.unparse(node)
    except Exception:
        return type(node).__name__


def _python_call_name(call: ast.Call) -> str:
    if isinstance(call.func, ast.Name):
        return call.func.id
    if isinstance(call.func, ast.Attribute):
        parts = []
        cur = call.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        return ".".join(reversed(parts))
    return "<call>"


# ---------------- Java ----------------

def _extract_java_method_blocks(source: str):
    lines = source.splitlines()
    results = []

    method_sig_re = re.compile(
        r'^\s*(?:public|protected|private|static|final|native|synchronized|abstract|default|strictfp|\s)+'
        r'.*?([A-Za-z_][A-Za-z0-9_]*)\s*\([^;]*\)\s*(?:throws\s+[^{]+)?\{?\s*$'
    )

    i = 0
    while i < len(lines):
        line = lines[i]
        m = method_sig_re.match(line)
        if not m:
            i += 1
            continue

        method_name = m.group(1)
        signature = line.rstrip()

        brace_line = i
        while brace_line < len(lines) and "{" not in lines[brace_line]:
            brace_line += 1
        if brace_line >= len(lines):
            i += 1
            continue

        depth = 0
        block_start = brace_line
        block_end = brace_line

        for j in range(brace_line, len(lines)):
            depth += lines[j].count("{")
            depth -= lines[j].count("}")
            if depth == 0:
                block_end = j
                break

        block_lines = lines[block_start:block_end + 1]
        results.append((method_name, signature, block_start + 1, block_lines))
        i = block_end + 1

    return results


def _java_parameters(signature: str) -> List[Dict[str, Any]]:
    m = re.search(r'\((.*)\)', signature)
    if not m:
        return []
    raw = m.group(1).strip()
    if not raw:
        return []

    out = []
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    for p in parts:
        toks = p.split()
        if len(toks) >= 2:
            out.append({"name": toks[-1], "type": " ".join(toks[:-1])})
        else:
            out.append({"name": p, "type": None})
    return out


def _java_call_hints(block_lines: List[str], start_line: int) -> List[Dict[str, Any]]:
    out = []
    for idx, raw in enumerate(block_lines):
        code = raw.strip()
        line_no = start_line + idx

        for m in re.finditer(r'\b([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)', code):
            callee = m.group(1)
            if callee in {"if", "for", "while", "switch", "catch", "return", "new", "throw", "super", "this"}:
                continue
            arg_text = m.group(2).strip()
            args = [a.strip() for a in arg_text.split(",")] if arg_text else []
            out.append(
                {
                    "line_no": line_no,
                    "callee": callee,
                    "arguments": args,
                }
            )
    return out


def _java_variable_origins(block_lines: List[str], start_line: int) -> List[Dict[str, Any]]:
    out = []
    for idx, raw in enumerate(block_lines):
        code = raw.strip()
        line_no = start_line + idx

        m = re.match(r'^\s*(?:[A-Za-z_<>\[\]]+\s+)+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+);$', code)
        if m:
            out.append(
                {
                    "variable": m.group(1),
                    "line_no": line_no,
                    "origin": m.group(2).strip(),
                }
            )
            continue

        m2 = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+);$', code)
        if m2:
            out.append(
                {
                    "variable": m2.group(1),
                    "line_no": line_no,
                    "origin": m2.group(2).strip(),
                }
            )
    return out


def _java_return_hints(block_lines: List[str], start_line: int) -> List[Dict[str, Any]]:
    out = []
    for idx, raw in enumerate(block_lines):
        code = raw.strip()
        line_no = start_line + idx
        if code.startswith("return "):
            out.append(
                {
                    "line_no": line_no,
                    "value": code[len("return "):].rstrip(";").strip(),
                }
            )
        elif code == "return;":
            out.append(
                {
                    "line_no": line_no,
                    "value": None,
                }
            )
    return out


def _as_dict(obj: Any) -> Dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}