# src/beacon_system/agents/prompts.py
# -*- coding: utf-8 -*-

"""
Unified prompt templates for Beacon agent workflow.

Goals:
- Keep planning / generate / revise on the same contract surface.
- Force all prompts to consume Task + BeaconIR + Constraints + OutputFormat.
- Avoid prompt drift across modules.
- Keep templates deterministic and simple.

This module only builds prompt strings.
It does NOT call the model and does NOT parse model output.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Optional
import json

from ..types import (
    Constraints,
    OutputFormatSpec,
    TaskObject,
    ThoughtCandidate,
)


# ============================================================
# Small deterministic helpers
# ============================================================


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, default=str)


def _join_lines(parts: Iterable[str]) -> str:
    lines: List[str] = []
    for p in parts:
        if p is None:
            continue
        text = str(p).strip()
        if text:
            lines.append(text)
    return "\n\n".join(lines).strip()


def _limit_text(text: str, max_chars: int = 12000) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    return getattr(obj, name, default)


def _as_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    return {}


def _as_seq(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


# ============================================================
# Shared blocks
# ============================================================


def render_task_block(task: TaskObject) -> str:
    return _join_lines([
        "## Task Context",
        _stable_json({
            "id": task.id,
            "lang": task.lang,
            "level": task.level,
            "target": task.target,
            "spec": task.spec,
            "context": task.context,
            "meta": task.meta,
        }),
    ])


def render_target_contract_block(task: TaskObject) -> str:
    return _join_lines([
        "## Target Contract",
        (
            "You are working on exactly one target.\n"
            f"- file: {task.target.get('file', '')}\n"
            f"- qualname: {task.target.get('qualname', '')}\n"
            f"- language: {task.lang}\n"
            f"- level: {task.level}"
        ),
        (
            "You must stay aligned with the target contract.\n"
            "Do not change unrelated symbols or redesign the module unless explicitly required by the task."
        ),
    ])


def _compact_ir(ir: Any) -> dict:
    entry = _as_dict(_get_attr(ir, "entry", {}))
    symbols = _get_attr(ir, "symbols", None)

    compact_symbols = {
        "imports": list(_get_attr(symbols, "imports", ()) or ()),
        "globals": list(_get_attr(symbols, "globals", ()) or ()),
        "attrs": list(_get_attr(symbols, "attrs", ()) or ()),
        "calls": list(_get_attr(symbols, "calls", ()) or ()),
    }

    nodes = []
    for n in _as_seq(_get_attr(ir, "nodes", ())):
        anchor = _get_attr(n, "anchor", None)
        nodes.append({
            "id": _get_attr(n, "id", ""),
            "kind": _get_attr(n, "kind", ""),
            "text": _get_attr(n, "text", ""),
            "anchor": {
                "file": _get_attr(anchor, "file", ""),
                "qualname": _get_attr(anchor, "qualname", ""),
                "lineno": _get_attr(anchor, "lineno", ""),
                "end_lineno": _get_attr(anchor, "end_lineno", ""),
                "namespace": _get_attr(anchor, "namespace", ""),
            },
            "meta": _get_attr(n, "meta", {}) or {},
        })

    edges = []
    for e in _as_seq(_get_attr(ir, "edges", ())):
        edges.append({
            "kind": _get_attr(e, "kind", ""),
            "src": _get_attr(e, "src", ""),
            "dst": _get_attr(e, "dst", ""),
            "meta": _get_attr(e, "meta", {}) or {},
        })

    return {
        "version": _get_attr(ir, "version", None),
        "entry": entry,
        "symbols": compact_symbols,
        "forbidden": list(_get_attr(ir, "forbidden", ()) or ()),
        "meta": _get_attr(ir, "meta", {}) or {},
        "nodes": nodes,
        "edges": edges,
    }


def _compact_constraints(constraints: Any) -> dict:
    return {
        "version": _get_attr(constraints, "version", None),
        "required_symbols": list(_get_attr(constraints, "required_symbols", ()) or ()),
        "required_calls": list(_get_attr(constraints, "required_calls", ()) or ()),
        "forbidden_specs": list(_get_attr(constraints, "forbidden_specs", ()) or ()),
        "match_specs": list(_get_attr(constraints, "match_specs", ()) or ()),
        "meta": _get_attr(constraints, "meta", {}) or {},
    }


def render_beacon_block(ir: Any, constraints: Constraints) -> str:
    return _join_lines([
        "## Beacon Block",
        (
            "The following Beacon IR and Constraints are the authoritative reasoning artifacts.\n"
            "You must use them directly.\n"
            "Do not ignore them, and do not reconstruct a separate reasoning source."
        ),
        "### Beacon IR",
        _limit_text(_stable_json(_compact_ir(ir))),
        "### Constraints",
        _limit_text(_stable_json(_compact_constraints(constraints))),
    ])


def render_output_contract_block(
    fmt: Optional[OutputFormatSpec],
    *,
    mode: str,
) -> str:
    fmt = fmt or OutputFormatSpec()

    if mode == "generate":
        rules = [
            "Return code only.",
            "Do not add explanations before or after the code.",
            "Do not wrap the code in markdown fences unless explicitly allowed.",
            "The returned text must be directly patchable into the target implementation.",
        ]
        if not fmt.fenced_code_block:
            rules.append("Markdown code fences are forbidden.")
        if fmt.require_language_match:
            rules.append("The code language must match the task language.")
        if fmt.single_block_only:
            rules.append("Return a single primary code block only.")
        return _join_lines([
            "## Output Contract",
            "\n".join(f"- {x}" for x in rules),
        ])

    if mode == "planning":
        return _join_lines([
            "## Output Contract",
            (
                "Return a compact planning result as plain text.\n"
                "Do not output final code.\n"
                "Do not output markdown fences.\n"
                "Each candidate should describe an implementation path, not prose discussion."
            ),
            (
                "Expected structure:\n"
                "THOUGHT 1:\n"
                "Rationale: ...\n"
                "Steps:\n"
                "- ...\n"
                "- ...\n"
                "Assumptions:\n"
                "- ...\n"
            ),
        ])

    if mode == "revise":
        rules = [
            "Return revised code only.",
            "Do not explain the fix outside the code.",
            "Apply only the minimum changes required by verifier/directives/runtime evidence.",
        ]
        if not fmt.fenced_code_block:
            rules.append("Markdown code fences are forbidden.")
        return _join_lines([
            "## Output Contract",
            "\n".join(f"- {x}" for x in rules),
        ])

    raise ValueError(f"unsupported output contract mode: {mode}")


def render_memory_block(memory_text: Optional[str]) -> str:
    if not memory_text or not str(memory_text).strip():
        return _join_lines([
            "## Experience Memory",
            "No useful prior memory is available for this task.",
        ])

    return _join_lines([
        "## Experience Memory",
        _limit_text(str(memory_text).strip(), max_chars=6000),
    ])


def render_selected_thought_block(thought: Optional[ThoughtCandidate]) -> str:
    if thought is None:
        return _join_lines([
            "## Selected Thought",
            "No selected thought is provided.",
        ])

    return _join_lines([
        "## Selected Thought",
        _stable_json({
            "id": _get_attr(thought, "id", ""),
            "text": _get_attr(thought, "text", ""),
            "rationale": _get_attr(thought, "rationale", ""),
            "steps": list(_get_attr(thought, "steps", ()) or ()),
            "assumptions": list(_get_attr(thought, "assumptions", ()) or ()),
            "meta": _get_attr(thought, "meta", {}) or {},
        }),
    ])


def render_revision_evidence_block(
    *,
    previous_code: str,
    verifier_summary: Optional[Any] = None,
    runtime_summary: Optional[Any] = None,
    beacon_usage_summary: Optional[Any] = None,
) -> str:
    return _join_lines([
        "## Previous Attempt",
        _limit_text(previous_code, max_chars=12000),
        "## Revision Evidence",
        _stable_json({
            "verifier": verifier_summary,
            "runtime": runtime_summary,
            "beacon_usage": beacon_usage_summary,
        }),
    ])


# ============================================================
# System prompts
# ============================================================


def planning_system_prompt() -> str:
    return _join_lines([
        "You are the planning stage of a Beacon-constrained code-generation system.",
        (
            "Your job is to propose a small number of implementation thoughts using the provided "
            "Task, Beacon IR, Constraints, and optional experience memory."
        ),
        (
            "You are not allowed to ignore Beacon artifacts, invent unrelated redesigns, "
            "or output final code."
        ),
        (
            "Prefer simple, stable, target-aligned solutions that satisfy the contract with minimal risk."
        ),
    ])


def generate_system_prompt() -> str:
    return _join_lines([
        "You are the code generation stage of a Beacon-constrained code-generation system.",
        (
            "Your job is to produce the target implementation by following the provided "
            "Task, Beacon IR, Constraints, and selected thought."
        ),
        (
            "You must preserve target alignment, obey required symbols/calls, avoid forbidden specs, "
            "and return patchable code only."
        ),
        "Do not add explanations, markdown fences, or unrelated refactors.",
    ])


def revise_system_prompt() -> str:
    return _join_lines([
        "You are the revision stage of a Beacon-constrained code-generation system.",
        (
            "Your job is to minimally revise a previous implementation using verifier feedback, "
            "runtime evidence, Beacon usage checks, Task, Beacon IR, Constraints, and the selected thought."
        ),
        (
            "Prioritize fixing issues that are likely to cause runtime errors, incorrect API usage, "
            "name-resolution failures, constructor or method misuse, or test failures."
        ),
        (
            "When revising, strictly respect the target file context and in-scope symbols. "
            "Do not assume missing imports, modules, helper functions, or global names exist unless they are provided in context."
        ),
        (
            "Do not introduce fake helper calls, fake global functions, detached intermediate variables, "
            "or placeholder uses of required symbols or calls."
        ),
        (
            "Every required symbol or call that appears in the code must participate in a valid execution path "
            "and must be used in a way that is consistent with the file context and the target API."
        ),
        (
            "Do not add dead code or unused variables just to satisfy Beacon usage checks."
        ),
        (
            "Preserve the current approach when possible, and make the smallest changes needed to satisfy "
            "Beacon constraints and improve execution correctness."
        ),
        (
            "Do not redesign the task from scratch unless the evidence shows the current approach is invalid."
        ),
        (
            "Do not make stylistic or unnecessary refactors. Focus only on corrections that materially improve "
            "correctness, executability, or constraint satisfaction."
        ),
        "Return corrected code only.",
    ])

# ============================================================
# User prompts
# ============================================================


def build_planning_prompt(
    *,
    task: TaskObject,
    ir: Any,
    constraints: Constraints,
    memory_text: Optional[str] = None,
    output_format: Optional[OutputFormatSpec] = None,
    max_thoughts: int = 3,
) -> str:
    return _join_lines([
        render_task_block(task),
        render_target_contract_block(task),
        render_beacon_block(ir, constraints),
        render_memory_block(memory_text),
        render_output_contract_block(output_format, mode="planning"),
        _join_lines([
            "## Planning Objective",
            (
                f"Generate up to {max_thoughts} distinct implementation thoughts.\n"
                "Each thought must be practical, minimal, and grounded in the Beacon artifacts.\n"
                "Prefer solutions that are simple to patch and easy to verify."
            ),
        ]),
    ])


def build_generate_prompt(
    *,
    task: TaskObject,
    ir: Any,
    constraints: Constraints,
    selected_thought: Optional[ThoughtCandidate],
    output_format: Optional[OutputFormatSpec] = None,
    extra_instructions: Optional[str] = None,
) -> str:
    return _join_lines([
        render_task_block(task),
        render_target_contract_block(task),
        render_beacon_block(ir, constraints),
        render_selected_thought_block(selected_thought),
        render_output_contract_block(output_format, mode="generate"),
        _join_lines([
            "## Generation Objective",
            (
                "Produce the target implementation now.\n"
                "The result must satisfy the task contract and be directly usable by the patcher/runtime."
            ),
        ]),
        _join_lines([
            "## Additional Instructions",
            str(extra_instructions).strip() if extra_instructions else "No additional instructions.",
        ]),
    ])


def build_revise_prompt(
    *,
    task: TaskObject,
    ir: Any,
    constraints: Constraints,
    selected_thought: Optional[ThoughtCandidate],
    previous_code: str,
    verifier_summary: Optional[Any] = None,
    runtime_summary: Optional[Any] = None,
    beacon_usage_summary: Optional[Any] = None,
    output_format: Optional[OutputFormatSpec] = None,
    extra_instructions: Optional[str] = None,
) -> str:
    return _join_lines([
        render_task_block(task),
        render_target_contract_block(task),
        render_beacon_block(ir, constraints),
        render_selected_thought_block(selected_thought),
        render_revision_evidence_block(
            previous_code=previous_code,
            verifier_summary=verifier_summary,
            runtime_summary=runtime_summary,
            beacon_usage_summary=beacon_usage_summary,
        ),
        render_output_contract_block(output_format, mode="revise"),
        _join_lines([
            "## Revision Objective",
            (
                "Revise the previous code with the minimum necessary changes.\n"
                "Fix only what is required by the evidence and constraints."
            ),
        ]),
        _join_lines([
            "## Additional Instructions",
            str(extra_instructions).strip() if extra_instructions else "No additional instructions.",
        ]),
    ])


# ============================================================
# Convenience message builders
# ============================================================


def make_planning_messages(
    *,
    task: TaskObject,
    ir: Any,
    constraints: Constraints,
    memory_text: Optional[str] = None,
    output_format: Optional[OutputFormatSpec] = None,
    max_thoughts: int = 3,
) -> List[dict]:
    return [
        {"role": "system", "content": planning_system_prompt()},
        {
            "role": "user",
            "content": build_planning_prompt(
                task=task,
                ir=ir,
                constraints=constraints,
                memory_text=memory_text,
                output_format=output_format,
                max_thoughts=max_thoughts,
            ),
        },
    ]


def make_generate_messages(
    *,
    task: TaskObject,
    ir: Any,
    constraints: Constraints,
    selected_thought: Optional[ThoughtCandidate],
    output_format: Optional[OutputFormatSpec] = None,
    extra_instructions: Optional[str] = None,
) -> List[dict]:
    return [
        {"role": "system", "content": generate_system_prompt()},
        {
            "role": "user",
            "content": build_generate_prompt(
                task=task,
                ir=ir,
                constraints=constraints,
                selected_thought=selected_thought,
                output_format=output_format,
                extra_instructions=extra_instructions,
            ),
        },
    ]


def make_revise_messages(
    *,
    task: TaskObject,
    ir: Any,
    constraints: Constraints,
    selected_thought: Optional[ThoughtCandidate],
    previous_code: str,
    verifier_summary: Optional[Any] = None,
    runtime_summary: Optional[Any] = None,
    beacon_usage_summary: Optional[Any] = None,
    output_format: Optional[OutputFormatSpec] = None,
    extra_instructions: Optional[str] = None,
) -> List[dict]:
    return [
        {"role": "system", "content": revise_system_prompt()},
        {
            "role": "user",
            "content": build_revise_prompt(
                task=task,
                ir=ir,
                constraints=constraints,
                selected_thought=selected_thought,
                previous_code=previous_code,
                verifier_summary=verifier_summary,
                runtime_summary=runtime_summary,
                beacon_usage_summary=beacon_usage_summary,
                output_format=output_format,
                extra_instructions=extra_instructions,
            ),
        },
    ]