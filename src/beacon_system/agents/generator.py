# src/beacon_system/agents/generator.py
# -*- coding: utf-8 -*-

"""
Code Generator Agent (Beacon-constrained)

Design goals:
- Consume ONLY: TaskObject + BeaconIR + Constraints + injected LLMClient (+ optional memory)
- NO Beacon reasoning, NO callgraph inference, NO env access, NO adapter/runtime coupling
- Produce code that satisfies Constraints (required_calls/symbols, forbidden_specs, match_specs)
- Deterministic prompt assembly (stable JSON, sorted lists) to support reproducible experiments

IMPORTANT:
- logic outputs (BeaconIR / Constraints) may be dataclass-like OR dict-like, and may not have `version`.
- This generator must be schema-tolerant (duck typing).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json

from ..types import TaskObject, Directive
from ..llm.client import LLMClient


# ----------------------------
# Prompt building (deterministic)
# ----------------------------

def _stable_json(obj: Any) -> str:
    """
    Local stable JSON for prompt assembly only.
    Canonical stable_json for artifacts is in beacon_system.io.stable_json.
    """
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sorted_unique(xs: Iterable[str]) -> List[str]:
    return sorted(set([x for x in xs if x is not None and str(x).strip() != ""]))


def _clip_text(s: str, limit: int) -> str:
    s = s or ""
    if len(s) <= limit:
        return s
    return s[:limit] + "\n# ... (truncated)\n"


def _format_match_specs(specs: Sequence[Any], limit: int = 60) -> List[Dict[str, Any]]:
    """
    MatchSpec is required to be stable_json serializable by contract.
    Keep a small cap to avoid prompt bloat.
    """
    out: List[Dict[str, Any]] = []
    for sp in list(specs)[:limit]:
        if isinstance(sp, dict):
            out.append(sp)
        else:
            d = getattr(sp, "__dict__", None)
            out.append(d if isinstance(d, dict) else {"repr": repr(sp)})
    return out


@dataclass(frozen=True)
class GeneratorOptions:
    max_ir_nodes: int = 80
    max_ir_edges: int = 120
    max_spec_chars: int = 4000
    max_prev_code_chars: int = 6000
    enforce_no_markdown: bool = True


class CodeGenerator:
    """
    Minimal generator agent; prompt compiler + LLM call wrapper.

    memory: optional object that MAY support:
      - retrieve_project(task: TaskObject) -> dict | None
      - retrieve_experience(task: TaskObject, ir: Any, constraints: Any) -> dict | None
    These are strictly optional and treated as passive context.
    """

    def __init__(self, llm: LLMClient, *, options: Optional[GeneratorOptions] = None):
        self._llm = llm
        self._opt = options or GeneratorOptions()

    # ----------------------------
    # Public API
    # ----------------------------

    def generate(
        self,
        task: TaskObject,
        ir: Any,
        constraints: Any,
        memory: Optional[object] = None,
    ) -> str:
        messages = self._build_generate_messages(task, ir, constraints, memory)
        code = self._llm.chat(messages)
        return self._postprocess_code(code)

    def revise(
        self,
        task: TaskObject,
        ir: Any,
        constraints: Any,
        llm: LLMClient,  # allow override
        directives: Tuple[Directive, ...],
        prev_code: str,
        memory: Optional[object] = None,
    ) -> str:
        messages = self._build_revise_messages(task, ir, constraints, directives, prev_code, memory)
        code = llm.chat(messages)
        return self._postprocess_code(code)

    # ----------------------------
    # Prompt templates
    # ----------------------------

    def _system_prompt(self) -> str:
        lines = [
            "You are a code generation agent in a Beacon-constrained pipeline.",
            "Beacon Logic is executed upstream. You MUST NOT re-derive reasoning rules.",
            "Your job: implement the target specified by TaskObject while satisfying Constraints.",
            "",
            "Hard rules:",
            "1) Satisfy required_calls and required_symbols explicitly in the implementation.",
            "2) Do NOT introduce forbidden patterns (forbidden_specs).",
            "3) Use only grounded symbols/calls from context or the task/project.",
            "4) Output must be Python code ONLY, no markdown, no explanations.",
            "5) Prefer minimal, stable, readable implementation; avoid over-engineering.",
        ]
        return "\n".join(lines)

    def _build_task_payload(self, task: TaskObject) -> Dict[str, Any]:
        return {
            "id": task.id,
            "lang": task.lang,
            "level": task.level,
            "target": {"file": task.target.get("file"), "qualname": task.target.get("qualname")},
            "spec": _clip_text(task.spec or "", self._opt.max_spec_chars),
            "context_keys": _sorted_unique(list((task.context or {}).keys())),
            "meta_keys": _sorted_unique(list((task.meta or {}).keys())),
        }

    # ----------------------------
    # Schema-tolerant IR/Constraints payloads
    # ----------------------------

    def _build_ir_payload(self, ir: Any) -> Dict[str, Any]:
        """
        Build a minimal IR payload for prompting.

        Supports:
        - dataclass-like IR (attributes)
        - dict-like IR (keys)
        - missing `version` (infer from meta)
        - symbols as dict or dataclass
        """
        def g(obj: Any, name: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        nodes_raw = g(ir, "nodes", []) or []
        edges_raw = g(ir, "edges", []) or []

        nodes = list(nodes_raw)[: self._opt.max_ir_nodes]
        edges = list(edges_raw)[: self._opt.max_ir_edges]

        def slim_anchor(a: Any) -> Dict[str, Any]:
            if isinstance(a, dict):
                return {
                    "file": a.get("file"),
                    "qualname": a.get("qualname"),
                    "lineno": a.get("lineno"),
                    "end_lineno": a.get("end_lineno"),
                    "namespace": a.get("namespace"),
                }
            return {
                "file": getattr(a, "file", None),
                "qualname": getattr(a, "qualname", None),
                "lineno": getattr(a, "lineno", None),
                "end_lineno": getattr(a, "end_lineno", None),
                "namespace": getattr(a, "namespace", None),
            }

        node_slim: List[Dict[str, Any]] = []
        for n in nodes:
            if isinstance(n, dict):
                node_slim.append(
                    {
                        "id": n.get("id"),
                        "kind": n.get("kind"),
                        "text": n.get("text"),
                        "anchor": slim_anchor(n.get("anchor", {})),
                        "meta": n.get("meta") or {},
                    }
                )
            else:
                a = getattr(n, "anchor", None)
                node_slim.append(
                    {
                        "id": getattr(n, "id", None),
                        "kind": getattr(n, "kind", None),
                        "text": getattr(n, "text", None),
                        "anchor": slim_anchor(a or {}),
                        "meta": getattr(n, "meta", None) or {},
                    }
                )

        edge_slim: List[Dict[str, Any]] = []
        for e in edges:
            if isinstance(e, dict):
                edge_slim.append(
                    {"kind": e.get("kind"), "src": e.get("src"), "dst": e.get("dst"), "meta": e.get("meta") or {}}
                )
            else:
                edge_slim.append(
                    {
                        "kind": getattr(e, "kind", None),
                        "src": getattr(e, "src", None),
                        "dst": getattr(e, "dst", None),
                        "meta": getattr(e, "meta", None) or {},
                    }
                )

        sym = g(ir, "symbols", {}) or {}
        if isinstance(sym, dict):
            imports = list(sym.get("imports", []) or [])
            globs = list(sym.get("globals", []) or [])
            attrs = list(sym.get("attrs", []) or [])
            calls = list(sym.get("calls", []) or [])
        else:
            imports = list(getattr(sym, "imports", ()) or ())
            globs = list(getattr(sym, "globals", ()) or ())
            attrs = list(getattr(sym, "attrs", ()) or ())
            calls = list(getattr(sym, "calls", ()) or ())

        meta = g(ir, "meta", {}) or {}
        entry = g(ir, "entry", {}) or {}

        version = g(ir, "version", None)
        if version is None:
            version = meta.get("schema_version") or meta.get("version") or "mvp-0.1"

        forbidden = g(ir, "forbidden", []) or []

        return {
            "version": version,
            "entry": entry,
            "nodes": node_slim,
            "edges": edge_slim,
            "symbols": {"imports": imports, "globals": globs, "attrs": attrs, "calls": calls},
            "forbidden_node_ids": list(forbidden),
            "meta": meta,
        }

    def _build_constraints_payload(self, constraints: Any) -> Dict[str, Any]:
        """
        Constraints payload builder (schema-tolerant).

        Supports:
        - dataclass-like constraints (attributes)
        - dict-like constraints (keys)
        - missing `version` (infer from meta)
        """
        def g(obj: Any, name: str, default: Any = None) -> Any:
            if isinstance(obj, dict):
                return obj.get(name, default)
            return getattr(obj, name, default)

        meta = g(constraints, "meta", {}) or {}

        version = g(constraints, "version", None)
        if version is None:
            version = (
                meta.get("constraints_version")
                or meta.get("schema_version")
                or meta.get("version")
                or "mvp-0.1"
            )

        required_symbols = g(constraints, "required_symbols", ()) or ()
        required_calls = g(constraints, "required_calls", ()) or ()
        forbidden_specs = g(constraints, "forbidden_specs", ()) or ()
        match_specs = g(constraints, "match_specs", ()) or ()

        return {
            "version": version,
            "required_symbols": _sorted_unique(required_symbols),
            "required_calls": _sorted_unique(required_calls),
            "forbidden_specs": _format_match_specs(forbidden_specs),
            "match_specs": _format_match_specs(match_specs),
            "meta": meta,
        }

    # ----------------------------
    # Optional memory retrieval hooks
    # ----------------------------

    def _maybe_retrieve_memory(self, memory: Optional[object], task: TaskObject, ir: Any, constraints: Any) -> Dict[str, Any]:
        if memory is None:
            return {}

        out: Dict[str, Any] = {}
        retrieve_project = getattr(memory, "retrieve_project", None)
        if callable(retrieve_project):
            try:
                out["project_memory"] = retrieve_project(task) or {}
            except Exception:
                out["project_memory"] = {}

        retrieve_experience = getattr(memory, "retrieve_experience", None)
        if callable(retrieve_experience):
            try:
                out["experience_memory"] = retrieve_experience(task, ir, constraints) or {}
            except Exception:
                out["experience_memory"] = {}
        return out

    # ----------------------------
    # Message builders
    # ----------------------------

    def _build_generate_messages(self, task: TaskObject, ir: Any, constraints: Any, memory: Optional[object]) -> List[Dict[str, str]]:
        task_payload = self._build_task_payload(task)
        ir_payload = self._build_ir_payload(ir)
        cons_payload = self._build_constraints_payload(constraints)
        mem_payload = self._maybe_retrieve_memory(memory, task, ir, constraints)

        user_payload = {
            "task": task_payload,
            "beacon_ir": ir_payload,
            "constraints": cons_payload,
            "memory": mem_payload,
            "instructions": [
                "Implement the target qualname in the target file.",
                "Satisfy required_calls and required_symbols explicitly.",
                "Avoid forbidden_specs strictly.",
                "Return Python code ONLY.",
            ],
        }
        return [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": _stable_json(user_payload)},
        ]

    def _build_revise_messages(
        self,
        task: TaskObject,
        ir: Any,
        constraints: Any,
        directives: Tuple[Directive, ...],
        prev_code: str,
        memory: Optional[object],
    ) -> List[Dict[str, str]]:
        task_payload = self._build_task_payload(task)
        ir_payload = self._build_ir_payload(ir)
        cons_payload = self._build_constraints_payload(constraints)
        mem_payload = self._maybe_retrieve_memory(memory, task, ir, constraints)

        dir_payload: List[Dict[str, Any]] = []
        for d in directives:
            dp = getattr(d, "__dict__", None)
            dir_payload.append(dp if isinstance(dp, dict) else {"repr": repr(d)})

        user_payload = {
            "task": task_payload,
            "beacon_ir": ir_payload,
            "constraints": cons_payload,
            "memory": mem_payload,
            "directives": dir_payload,
            "prev_code": _clip_text(prev_code or "", self._opt.max_prev_code_chars),
            "instructions": [
                "Revise the previous code according to directives.",
                "Do NOT break existing correct behavior.",
                "Ensure required_calls/required_symbols coverage improves or stays satisfied.",
                "Avoid forbidden_specs strictly.",
                "Return Python code ONLY.",
            ],
        }
        return [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": _stable_json(user_payload)},
        ]

    # ----------------------------
    # Output post-processing
    # ----------------------------

    def _postprocess_code(self, raw: str) -> str:
        s = (raw or "").strip()

        # Strip markdown fences if the model violated format.
        if s.startswith("```"):
            parts = s.split("\n")
            if parts and parts[0].startswith("```"):
                parts = parts[1:]
            if parts and parts[-1].strip().startswith("```"):
                parts = parts[:-1]
            s = "\n".join(parts).strip()

        if self._opt.enforce_no_markdown:
            idx = s.find("def ")
            if idx > 0:
                prefix = s[:idx].lower()
                if any(tok in prefix for tok in ["explain", "here", "note", "说明", "当然"]):
                    s = s[idx:].lstrip()

        return s


# ----------------------------
# Contract-aligned functional wrappers
# ----------------------------

def generate(
    task: TaskObject,
    ir: Any,
    constraints: Any,
    llm: LLMClient,
    memory: Optional[object] = None,
) -> str:
    return CodeGenerator(llm).generate(task, ir, constraints, memory)


def revise(
    task: TaskObject,
    ir: Any,
    constraints: Any,
    llm: LLMClient,
    directives: Tuple[Directive, ...],
    prev_code: str,
    memory: Optional[object] = None,
) -> str:
    gen = CodeGenerator(llm)
    return gen.revise(task, ir, constraints, llm, directives, prev_code, memory)