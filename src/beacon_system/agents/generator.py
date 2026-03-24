# -*- coding: utf-8 -*-

"""
Code Generator Agent (strict code-only mode)

Design goals:
- Generate code only
- No planning / no scoring / no verification
- Consume only structured task inputs + LLM client
- Prompt must explicitly forbid explanations and extra text
- Keep implementation simple, deterministic, and easy to debug

Accepted inputs:
- task context
- target file / target function
- beacon_tree
- signature_hints
- constraint_summary
- injected LLMClient

Output:
- generated_code
- prompt_snapshot
- raw_response
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional
import json
import re

from ..llm.client import LLMClient


# ============================================================
# helpers
# ============================================================

def _stable_json(obj: Any) -> str:
    """
    Stable JSON string for prompt construction and artifact trace.
    """
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _as_dict(obj: Any) -> Dict[str, Any]:
    """
    Tolerant object-to-dict conversion.

    Supports:
    - dict
    - dataclass-like objects
    - normal objects with __dict__
    - fallback to {"value": str(obj)}
    """
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dataclass_fields__"):
        try:
            return asdict(obj)
        except Exception:
            pass
    if hasattr(obj, "__dict__"):
        try:
            return dict(vars(obj))
        except Exception:
            pass
    return {"value": str(obj)}


def _compact_task_context(task: Any) -> Dict[str, Any]:
    """
    Extract only the generator-relevant task fields.
    Keep it concise to reduce prompt noise.
    """
    data = _as_dict(task)

    return {
        "task_id": data.get("task_id"),
        "lang": data.get("lang"),
        "target_file": data.get("target_file"),
        "target_function": data.get("target_function"),
        "signature": data.get("signature"),
        "docstring": data.get("docstring"),
        "instruction": data.get("instruction"),
        "prompt": data.get("prompt"),
        "file_path": data.get("file_path"),
        "function_name": data.get("function_name"),
        "class_name": data.get("class_name"),
        "code_context": data.get("code_context"),
        "context_blocks": data.get("context_blocks"),
    }


def _extract_target_file(task: Any) -> Optional[str]:
    data = _as_dict(task)
    for key in ("target_file", "file_path"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_target_function(task: Any) -> Optional[str]:
    data = _as_dict(task)
    for key in ("target_function", "function_name", "entry_function", "name"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences if the model still outputs them.
    """
    text = text.strip()

    fenced = re.match(r"^\s*```[a-zA-Z0-9_+-]*\s*\n(.*?)\n```\s*$", text, flags=re.DOTALL)
    if fenced:
        return fenced.group(1).strip()

    text = re.sub(r"^\s*```[a-zA-Z0-9_+-]*\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def _remove_obvious_non_code_prefix(text: str) -> str:
    """
    Best-effort cleanup for common model violations like:
    - 'Here is the code:'
    - 'Sure, here's the implementation'
    Keep logic conservative to avoid corrupting real code.
    """
    lines = text.strip().splitlines()
    if not lines:
        return ""

    bad_prefix_patterns = (
        r"^\s*here(?:'s| is)\b",
        r"^\s*sure\b",
        r"^\s*the implementation\b",
        r"^\s*implementation:\s*$",
        r"^\s*code:\s*$",
        r"^\s*generated code:\s*$",
    )

    while lines:
        first = lines[0].strip().lower()
        if any(re.match(p, first, flags=re.IGNORECASE) for p in bad_prefix_patterns):
            lines.pop(0)
            continue
        break

    return "\n".join(lines).strip()


def _sanitize_code_output(text: str) -> str:
    """
    Final code-only cleanup.
    """
    text = _strip_code_fences(text)
    text = _remove_obvious_non_code_prefix(text)
    return text.strip()


# ============================================================
# prompt builder
# ============================================================

def build_generator_system_prompt() -> str:
    """
    Strict system instruction:
    - code only
    - no explanations
    - no markdown
    """
    return (
        "You are a code generation agent.\n"
        "Your only job is to output the final code for the requested target.\n"
        "Do not explain anything.\n"
        "Do not add comments outside the code unless they are part of the code itself.\n"
        "Do not output markdown fences.\n"
        "Do not output analysis.\n"
        "Do not output bullet points.\n"
        "Do not output any text before or after the code.\n"
        "Output code only."
    )


def build_generator_user_prompt(
    *,
    task: Any,
    beacon_tree: Any,
    signature_hints: Any,
    constraint_summary: Any,
) -> str:
    """
    Build the strict code-generation prompt.
    """
    task_context = _compact_task_context(task)
    target_file = _extract_target_file(task)
    target_function = _extract_target_function(task)

    parts = [
        "Generate the target code using the structured inputs below.",
        "",
        "STRICT OUTPUT RULES:",
        "1. Output code only.",
        "2. Do not output markdown fences.",
        "3. Do not output explanations, notes, or prose.",
        "4. Do not describe what you changed.",
        "5. The output must be directly usable as code.",
        "6. Follow beacon_tree, signature_hints, and constraint_summary strictly.",
        "",
        "TARGET:",
        _stable_json(
            {
                "target_file": target_file,
                "target_function": target_function,
            }
        ),
        "",
        "TASK_CONTEXT:",
        _stable_json(task_context),
        "",
        "BEACON_TREE:",
        _stable_json(_as_dict(beacon_tree)),
        "",
        "SIGNATURE_HINTS:",
        _stable_json(_as_dict(signature_hints)),
        "",
        "CONSTRAINT_SUMMARY:",
        _stable_json(_as_dict(constraint_summary)),
        "",
        "FINAL REMINDER:",
        "Return code only. No markdown. No explanation. No surrounding text.",
    ]

    return "\n".join(parts).strip()


# ============================================================
# result objects
# ============================================================

@dataclass
class GeneratorInput:
    task: Any
    beacon_tree: Any
    signature_hints: Any
    constraint_summary: Any


@dataclass
class GeneratorResult:
    generated_code: str
    prompt_snapshot: str
    raw_response: Dict[str, Any]
    model_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# agent
# ============================================================

class CodeGeneratorAgent:
    """
    Strict code-only generator agent.

    Responsibilities:
    - Build generator prompt from structured inputs
    - Call LLM client once
    - Sanitize output into code-only form
    - Return artifacts for downstream workflow
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def run(
        self,
        *,
        task: Any,
        beacon_tree: Any,
        signature_hints: Any,
        constraint_summary: Any,
    ) -> GeneratorResult:
        """
        Run code generation once.

        No planning, no scoring, no verification.
        """
        system_prompt = build_generator_system_prompt()
        user_prompt = build_generator_user_prompt(
            task=task,
            beacon_tree=beacon_tree,
            signature_hints=signature_hints,
            constraint_summary=constraint_summary,
        )

        response = self.llm_client.chat(
            prompt=user_prompt,
            system_prompt=system_prompt,
        )

        code = _sanitize_code_output(response.text)

        return GeneratorResult(
            generated_code=code,
            prompt_snapshot=response.prompt_snapshot,
            raw_response=response.raw_response,
            model_name=getattr(response, "model_name", None),
        )