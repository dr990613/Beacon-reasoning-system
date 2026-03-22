# baseline_codegen/types.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import re


@dataclass
class CodeEvalTask:
    """
    Minimal task object for a CoderEval-style generation task.
    """
    task_id: str
    prompt: str
    signature: Optional[str] = None
    docstring: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_raw(cls, raw: Dict[str, Any]) -> "CodeEvalTask":
        """
        Build a task object from one raw record in the benchmark json.
        This method is schema-tolerant and tries several common field names.
        """
        task_id = str(
            raw.get("_id")
            or raw.get("question_id")
            or raw.get("task_id")
            or raw.get("id")
            or ""
        ).strip()

        signature = _clean_optional_text(
            raw.get("signature") or raw.get("prompt_signature")
        )
        docstring = _clean_optional_text(
            raw.get("docstring") or raw.get("comment")
        )

        prompt = build_task_prompt(
            name=_clean_optional_text(raw.get("name")),
            human_label=_clean_optional_text(raw.get("human_label")),
            signature=signature,
            docstring=docstring,
            file_path=_clean_optional_text(raw.get("file_path")),
            all_context=_clean_optional_text(raw.get("all_context")),
        )

        metadata = {
            "name": raw.get("name"),
            "file_path": raw.get("file_path"),
            "project": raw.get("project"),
            "package": raw.get("package"),
            "level": raw.get("level"),
            "raw": raw,
        }

        return cls(
            task_id=task_id,
            prompt=prompt,
            signature=signature,
            docstring=docstring,
            metadata=metadata,
        )


@dataclass
class ModelRequest:
    """
    Standard request object sent to the model layer.
    """
    system_prompt: str
    user_prompt: str
    temperature: float = 0.0
    max_tokens: int = 1024

    def to_chat_messages(self) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.user_prompt},
        ]


@dataclass
class ModelResponse:
    """
    Raw response returned from the model layer.
    """
    content: str
    finish_reason: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


@dataclass
class GenerationResult:
    """
    Final accepted generation result for one task.
    Output format is aligned to the required benchmark json style.
    """
    _id: str
    generate_results: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "_id": self._id,
            "generate_results": self.generate_results,
        }


@dataclass
class CodeOnlyPolicy:
    """
    Output policy: final accepted result must be code-like content only.
    """
    min_code_length: int = 10
    allow_markdown_fence: bool = True

    def normalize(self, text: str) -> str:
        """
        Normalize model output into plain code text.
        """
        text = (text or "").strip()
        if not text:
            return ""

        if self.allow_markdown_fence:
            fenced = extract_code_block(text)
            if fenced:
                text = fenced.strip()

        return text.strip()

    def is_code(self, text: str) -> bool:
        """
        Lightweight code check.
        Goal: reject obvious natural language answers, not to fully parse Python.
        """
        text = self.normalize(text)
        if len(text) < self.min_code_length:
            return False

        lowered = text.lower()

        obvious_nl_prefixes = (
            "here is",
            "here's",
            "the code",
            "this code",
            "solution:",
            "explanation:",
            "you can use",
            "sure,",
        )
        if lowered.startswith(obvious_nl_prefixes):
            return False

        code_signals = [
            r"^\s*def\s+\w+\s*\(",
            r"^\s*class\s+\w+\s*[\(:]",
            r"^\s*import\s+\w+",
            r"^\s*from\s+\w+\s+import\s+",
            r"^\s*if\s+__name__\s*==\s*['\"]__main__['\"]\s*:",
            r"\n\s*return\b",
            r"\n\s*for\b",
            r"\n\s*if\b",
            r"\n\s*try\s*:",
            r"\n\s*except\b",
            r"\n\s*with\b",
            r"\n\s*while\b",
            r"\n\s*@\w+",
            r":[ \t]*\n[ \t]+",
        ]
        if any(re.search(pattern, text, flags=re.MULTILINE) for pattern in code_signals):
            return True

        # fallback: strong punctuation/layout hints of code
        if "(" in text and ")" in text and ":" in text:
            return True
        if "=" in text and "\n" in text:
            return True

        return False

    def validate_or_raise(self, text: str) -> str:
        """
        Normalize and validate code output.
        """
        code = self.normalize(text)
        if not self.is_code(code):
            raise ValueError("Model output is not valid code-only content.")
        return code


def build_task_prompt(
    name: Optional[str],
    human_label: Optional[str],
    signature: Optional[str],
    docstring: Optional[str],
    file_path: Optional[str],
    all_context: Optional[str],
) -> str:
    """
    Build a stable prompt for baseline code generation.
    """
    parts: List[str] = []
    parts.append("Write Python code for the following task.")
    parts.append("Return only code. Do not include explanations.")

    if name:
        parts.append(f"Function name: {name}")
    if human_label:
        parts.append(f"Task: {human_label}")
    if signature:
        parts.append("Signature:")
        parts.append(signature)
    if docstring:
        parts.append("Docstring:")
        parts.append(docstring)
    if file_path:
        parts.append(f"Target file: {file_path}")
    if all_context:
        parts.append("Additional context:")
        parts.append(all_context)

    return "\n\n".join(parts).strip()


def extract_code_block(text: str) -> str:
    """
    Extract the first markdown code block if present.
    """
    if not text:
        return ""

    pattern = r"```(?:python|py)?\s*\n(.*?)```"
    match = re.search(pattern, text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _clean_optional_text(value: Any) -> Optional[str]:
    """
    Convert value to stripped text, returning None for empty values.
    """
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None