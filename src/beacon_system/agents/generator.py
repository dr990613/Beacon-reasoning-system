# src/beacon_system/agents/generator.py
# -*- coding: utf-8 -*-

"""
Code generator / reviser for Beacon agent workflow.

Scope:
- Consume ONLY:
  Task + BeaconIR + Constraints + selected thought + LLMClient
- Produce:
  GenerationPayload + FormatValidationResult
- Support:
  initial generation and minimal revision

Non-goals:
- no Beacon reasoning reconstruction
- no scoring
- no execution
- no verifier logic duplication
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple

from ..llm.client import LLMClient, LLMError
from ..types import (
    BeaconIR,
    CodeBlock,
    Constraints,
    FormatValidationResult,
    GenerationPayload,
    OutputFormatSpec,
    TaskObject,
    ThoughtCandidate,
)
from .prompts import make_generate_messages, make_revise_messages


def _safe_text(text: Optional[str]) -> str:
    return str(text or "")


def _guess_language(task: TaskObject) -> str:
    lang = str(task.lang or "").strip().lower()
    if lang:
        return lang
    return "python"


def _default_filename(task: TaskObject) -> str:
    return str(task.target.get("file") or "generated_code.txt")


def _extract_fenced_code(raw_text: str) -> Optional[str]:
    """
    Extract first fenced code block if present.
    Supports:
    ```python
    ...
    ```
    """
    text = _safe_text(raw_text)
    pattern = re.compile(r"```[a-zA-Z0-9_\-]*\n(.*?)```", re.DOTALL)
    m = pattern.search(text)
    if not m:
        return None
    return m.group(1).strip()


def _strip_common_explanations(raw_text: str) -> str:
    """
    Minimal cleanup when the model adds prose around code.

    Strategy:
    - prefer fenced block if present
    - otherwise drop a few common leading explanation lines
    - keep the rest untouched to avoid over-cleaning
    """
    text = _safe_text(raw_text).strip()
    if not text:
        return ""

    fenced = _extract_fenced_code(text)
    if fenced is not None:
        return fenced.strip()

    lines = text.splitlines()

    prefixes = (
        "here is",
        "here's",
        "below is",
        "the code",
        "updated code",
        "revised code",
        "solution:",
        "explanation:",
    )

    cleaned_lines = []
    skipping_prefix = True

    for line in lines:
        stripped = line.strip()
        lowered = stripped.lower()

        if skipping_prefix and stripped:
            if any(lowered.startswith(p) for p in prefixes):
                continue
            skipping_prefix = False

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def _validate_output_format(
    *,
    raw_text: str,
    normalized_code: str,
    task: TaskObject,
    fmt: Optional[OutputFormatSpec],
) -> FormatValidationResult:
    fmt = fmt or OutputFormatSpec()
    issues = []

    raw = _safe_text(raw_text)
    code = _safe_text(normalized_code)

    if not code.strip():
        issues.append("empty_code")

    if not fmt.fenced_code_block and "```" in raw:
        issues.append("markdown_fence_detected")

    if fmt.code_only:
        lowered = raw.lower()
        noisy_prefixes = (
            "here is",
            "here's",
            "below is",
            "the updated code",
            "the revised code",
            "explanation:",
        )
        if any(lowered.strip().startswith(p) for p in noisy_prefixes):
            issues.append("leading_explanation_detected")

    if fmt.require_language_match:
        lang = _guess_language(task)
        if lang == "python":
            # very light heuristic only
            if "class " not in code and "def " not in code and "=" not in code and "return" not in code:
                issues.append("language_match_weak")

    return FormatValidationResult(
        ok=len(issues) == 0,
        normalized_code=code,
        issues=tuple(issues),
        meta={
            "validator": "generator.format",
            "raw_len": len(raw),
            "code_len": len(code),
            "task_lang": task.lang,
        },
    )


def _build_generation_payload(
    *,
    raw_text: str,
    code: str,
    task: TaskObject,
    fmt_check: FormatValidationResult,
) -> GenerationPayload:
    primary = CodeBlock(
        language=_guess_language(task),
        content=code,
        kind="replacement_impl",
        filename=_default_filename(task),
        meta={
            "target_file": task.target.get("file"),
            "target_qualname": task.target.get("qualname"),
        },
    )
    return GenerationPayload(
        primary=primary,
        auxiliary=(),
        format_ok=fmt_check.ok,
        raw_text=_safe_text(raw_text),
        meta={
            "task_id": task.id,
            "target_file": task.target.get("file"),
            "target_qualname": task.target.get("qualname"),
        },
    )


@dataclass
class CodeGenerator:
    """
    Minimal Beacon-constrained generator.

    Responsibilities:
    - build generation/revision prompts
    - call LLM
    - normalize model output into GenerationPayload
    - run lightweight format validation

    Non-responsibilities:
    - no reasoning rebuild
    - no scoring
    - no execution
    - no verifier invocation
    """
    llm: LLMClient
    print_io: bool = False

    def _print(self, message: str) -> None:
        if self.print_io:
            print(f"[CodeGenerator] {message}")

    def _normalize_generation(
        self,
        *,
        raw_text: str,
        task: TaskObject,
        output_format: Optional[OutputFormatSpec],
    ) -> Tuple[GenerationPayload, FormatValidationResult]:
        cleaned = _strip_common_explanations(raw_text)
        fmt_check = _validate_output_format(
            raw_text=raw_text,
            normalized_code=cleaned,
            task=task,
            fmt=output_format,
        )
        payload = _build_generation_payload(
            raw_text=raw_text,
            code=fmt_check.normalized_code,
            task=task,
            fmt_check=fmt_check,
        )
        return payload, fmt_check

    def generate(
        self,
        *,
        task: TaskObject,
        ir: BeaconIR,
        constraints: Constraints,
        selected_thought: Optional[ThoughtCandidate],
        output_format: Optional[OutputFormatSpec] = None,
        extra_instructions: Optional[str] = None,
    ) -> Tuple[GenerationPayload, FormatValidationResult]:
        self._print(f"start generate: task={task.id}")

        messages = make_generate_messages(
            task=task,
            ir=ir,
            constraints=constraints,
            selected_thought=selected_thought,
            output_format=output_format,
            extra_instructions=extra_instructions,
        )

        try:
            raw_text = self.llm.generate_text(messages=messages)
            self._print(f"generate response chars={len(raw_text)}")
        except LLMError as e:
            self._print(f"generate llm error: {e}")
            raw_text = ""

        payload, fmt_check = self._normalize_generation(
            raw_text=raw_text,
            task=task,
            output_format=output_format,
        )

        self._print(
            f"generate done: format_ok={fmt_check.ok} code_len={len(fmt_check.normalized_code)}"
        )
        return payload, fmt_check

    def revise(
        self,
        *,
        task: TaskObject,
        ir: BeaconIR,
        constraints: Constraints,
        selected_thought: Optional[ThoughtCandidate],
        previous_code: str,
        verifier_summary: Optional[object] = None,
        runtime_summary: Optional[object] = None,
        beacon_usage_summary: Optional[object] = None,
        output_format: Optional[OutputFormatSpec] = None,
        extra_instructions: Optional[str] = None,
    ) -> Tuple[GenerationPayload, FormatValidationResult]:
        self._print(f"start revise: task={task.id}")

        messages = make_revise_messages(
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
        )

        try:
            raw_text = self.llm.generate_text(messages=messages)
            self._print(f"revise response chars={len(raw_text)}")
        except LLMError as e:
            self._print(f"revise llm error: {e}")
            raw_text = previous_code or ""

        payload, fmt_check = self._normalize_generation(
            raw_text=raw_text,
            task=task,
            output_format=output_format,
        )

        self._print(
            f"revise done: format_ok={fmt_check.ok} code_len={len(fmt_check.normalized_code)}"
        )
        return payload, fmt_check


def generate_code(
    *,
    llm: LLMClient,
    task: TaskObject,
    ir: BeaconIR,
    constraints: Constraints,
    selected_thought: Optional[ThoughtCandidate],
    output_format: Optional[OutputFormatSpec] = None,
    extra_instructions: Optional[str] = None,
    print_io: bool = False,
) -> Tuple[GenerationPayload, FormatValidationResult]:
    """
    Convenience function for initial generation.
    """
    generator = CodeGenerator(llm=llm, print_io=print_io)
    return generator.generate(
        task=task,
        ir=ir,
        constraints=constraints,
        selected_thought=selected_thought,
        output_format=output_format,
        extra_instructions=extra_instructions,
    )


def revise_code(
    *,
    llm: LLMClient,
    task: TaskObject,
    ir: BeaconIR,
    constraints: Constraints,
    selected_thought: Optional[ThoughtCandidate],
    previous_code: str,
    verifier_summary: Optional[object] = None,
    runtime_summary: Optional[object] = None,
    beacon_usage_summary: Optional[object] = None,
    output_format: Optional[OutputFormatSpec] = None,
    extra_instructions: Optional[str] = None,
    print_io: bool = False,
) -> Tuple[GenerationPayload, FormatValidationResult]:
    """
    Convenience function for revision generation.
    """
    generator = CodeGenerator(llm=llm, print_io=print_io)
    return generator.revise(
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
    )