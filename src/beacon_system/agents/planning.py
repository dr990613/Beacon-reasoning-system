# src/beacon_system/agents/planning.py
# -*- coding: utf-8 -*-

"""
Lightweight planning for Beacon agent workflow.

Scope:
- Generate a small number of implementation thoughts.
- Consume Task + BeaconIR + Constraints + optional memory + LLMClient.
- Output normalized ThoughtCandidate objects.

Non-goals:
- no complex tree-of-thought search
- no scoring/ranking here
- no code generation here
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..llm.client import LLMClient, LLMError
from ..types import BeaconIR, Constraints, OutputFormatSpec, TaskObject, ThoughtCandidate
from .prompts import make_planning_messages


def _safe_text(text: Optional[str]) -> str:
    return str(text or "").strip()


def _clean_line(text: str) -> str:
    return str(text or "").strip().lstrip("-").strip()


def _split_thought_blocks(text: str) -> List[str]:
    """
    Split model output into THOUGHT blocks.

    Expected planning prompt structure:
    THOUGHT 1:
    Rationale: ...
    Steps:
    - ...
    Assumptions:
    - ...

    This parser is intentionally tolerant.
    """
    text = _safe_text(text)
    if not text:
        return []

    pattern = re.compile(r"(?im)^\s*THOUGHT\s+\d+\s*:\s*")
    matches = list(pattern.finditer(text))

    if not matches:
        return [text]

    blocks: List[str] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        if block:
            blocks.append(block)
    return blocks


def _extract_section(block: str, label: str) -> str:
    """
    Extract a single-line or multi-line section body after a label.

    Example labels:
    - Rationale
    - Steps
    - Assumptions
    """
    block = str(block or "")
    pattern = re.compile(
        rf"(?ims)^\s*{re.escape(label)}\s*:\s*(.*?)"
        rf"(?=^\s*[A-Za-z][A-Za-z ]*\s*:|\Z)"
    )
    m = pattern.search(block)
    if not m:
        return ""
    return m.group(1).strip()


def _extract_steps(section_text: str) -> List[str]:
    """
    Parse bullet-like lines from Steps / Assumptions section.
    """
    section_text = _safe_text(section_text)
    if not section_text:
        return []

    lines = [line.rstrip() for line in section_text.splitlines()]
    items: List[str] = []

    for line in lines:
        raw = line.strip()
        if not raw:
            continue

        if raw.startswith(("-", "*", "•")):
            item = _clean_line(raw)
            if item:
                items.append(item)
            continue

        numbered = re.match(r"^\d+[\.\)]\s*(.+)$", raw)
        if numbered:
            item = _clean_line(numbered.group(1))
            if item:
                items.append(item)
            continue

        # Tolerant fallback: keep plain lines too
        items.append(raw)

    return items


def _extract_title_text(block: str) -> str:
    """
    Remove the THOUGHT n: header and keep the descriptive body.
    """
    block = str(block or "").strip()
    block = re.sub(r"(?im)^\s*THOUGHT\s+\d+\s*:\s*", "", block, count=1).strip()
    return block


def _make_fallback_thought(
    *,
    task: TaskObject,
    constraints: Constraints,
    reason: str,
) -> ThoughtCandidate:
    """
    Safe fallback when planning parse/model call fails.
    """
    steps: List[str] = [
        f"Edit target {task.target.get('file', '')}::{task.target.get('qualname', '')}.",
        "Keep the change minimal and target-aligned.",
    ]

    if constraints.required_symbols:
        steps.append(
            "Preserve/use required symbols: " + ", ".join(list(constraints.required_symbols)[:8])
        )
    if constraints.required_calls:
        steps.append(
            "Preserve/use required calls: " + ", ".join(list(constraints.required_calls)[:8])
        )

    return ThoughtCandidate(
        id="thought_fallback_1",
        text="Implement a minimal patch guided directly by Constraints and BeaconIR.",
        rationale=reason,
        steps=tuple(steps),
        assumptions=(
            "Constraints provide enough structure for a minimal implementation.",
        ),
        meta={"source": "fallback"},
    )


def _parse_thought_candidates(raw_text: str, max_thoughts: int) -> List[ThoughtCandidate]:
    """
    Parse model planning text into ThoughtCandidate objects.

    Tolerant behavior:
    - if no THOUGHT blocks, treat the whole output as one thought
    - if sections are missing, keep partial content
    """
    raw_text = _safe_text(raw_text)
    if not raw_text:
        return []

    blocks = _split_thought_blocks(raw_text)
    thoughts: List[ThoughtCandidate] = []

    for idx, block in enumerate(blocks[:max_thoughts], start=1):
        rationale = _extract_section(block, "Rationale")
        steps = _extract_steps(_extract_section(block, "Steps"))
        assumptions = _extract_steps(_extract_section(block, "Assumptions"))

        text_body = _extract_title_text(block)

        if not rationale and not steps and not assumptions:
            # very loose fallback parsing
            rationale = text_body

        thoughts.append(
            ThoughtCandidate(
                id=f"thought_{idx}",
                text=text_body,
                rationale=rationale,
                steps=tuple(steps),
                assumptions=tuple(assumptions),
                meta={"source": "llm"},
            )
        )

    return thoughts


@dataclass
class Planner:
    """
    Lightweight planner.

    Responsibilities:
    - build planning prompt/messages
    - call LLM
    - parse normalized ThoughtCandidate results
    - provide safe fallback when planning fails

    Non-responsibilities:
    - no scoring
    - no generation
    - no verification
    """
    llm: LLMClient
    max_thoughts: int = 3
    print_io: bool = False

    def _print(self, message: str) -> None:
        if self.print_io:
            print(f"[Planner] {message}")

    def plan(
        self,
        *,
        task: TaskObject,
        ir: BeaconIR,
        constraints: Constraints,
        memory_text: Optional[str] = None,
        output_format: Optional[OutputFormatSpec] = None,
    ) -> Sequence[ThoughtCandidate]:
        self._print(
            f"start planning: task={task.id} max_thoughts={self.max_thoughts}"
        )

        messages = make_planning_messages(
            task=task,
            ir=ir,
            constraints=constraints,
            memory_text=memory_text,
            output_format=output_format,
            max_thoughts=self.max_thoughts,
        )

        try:
            raw_text = self.llm.generate_text(messages=messages)
            self._print(f"llm planning response chars={len(raw_text)}")

            thoughts = _parse_thought_candidates(
                raw_text=raw_text,
                max_thoughts=self.max_thoughts,
            )

            if thoughts:
                self._print(f"planning parsed thoughts={len(thoughts)}")
                return tuple(thoughts)

            self._print("planning parse empty; using fallback thought")
            return (
                _make_fallback_thought(
                    task=task,
                    constraints=constraints,
                    reason="LLM planning output could not be parsed into valid thoughts.",
                ),
            )

        except LLMError as e:
            self._print(f"planning llm error: {e}")
            return (
                _make_fallback_thought(
                    task=task,
                    constraints=constraints,
                    reason=f"LLM planning failed: {e}",
                ),
            )
        except Exception as e:
            self._print(f"planning unexpected error: {e}")
            return (
                _make_fallback_thought(
                    task=task,
                    constraints=constraints,
                    reason=f"Unexpected planning failure: {e}",
                ),
            )


def generate_thoughts(
    *,
    llm: LLMClient,
    task: TaskObject,
    ir: BeaconIR,
    constraints: Constraints,
    memory_text: Optional[str] = None,
    output_format: Optional[OutputFormatSpec] = None,
    max_thoughts: int = 3,
    print_io: bool = False,
) -> Sequence[ThoughtCandidate]:
    """
    Convenience function for direct planning usage.
    """
    planner = Planner(
        llm=llm,
        max_thoughts=max_thoughts,
        print_io=print_io,
    )
    return planner.plan(
        task=task,
        ir=ir,
        constraints=constraints,
        memory_text=memory_text,
        output_format=output_format,
    )