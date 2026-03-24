# -*- coding: utf-8 -*-

"""
Beacon-based Verifier Agent

Responsibilities:
- Verify generated code against:
    - original beacon
    - rebuilt beacon
    - generated code
    - full task information / target
- Check:
    - required coverage
    - argument binding
    - dependency path consistency
    - return contract consistency
- Output structured result only
- No code generation
- Allow only one reflection pass

Design goals:
- Low-cost verification
- Strongly structured output
- Beacon-centered prompt
- Easy to connect after rebuilder
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional
import json

from ..llm.client import LLMClient


# ============================================================
# helpers
# ============================================================

def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _as_dict(obj: Any) -> Dict[str, Any]:
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
    data = _as_dict(task)
    return {
        "task_id": data.get("task_id"),
        "lang": data.get("lang"),
        "target_file": data.get("target_file") or data.get("file_path"),
        "target_function": (
            data.get("target_function")
            or data.get("function_name")
            or data.get("entry_function")
            or data.get("name")
        ),
        "signature": data.get("signature"),
        "docstring": data.get("docstring"),
        "instruction": data.get("instruction"),
        "prompt": data.get("prompt"),
        "class_name": data.get("class_name"),
        "lineno": data.get("lineno"),
        "end_lineno": data.get("end_lineno"),
    }


def _extract_json_block(text: str) -> Dict[str, Any]:
    """
    Best-effort JSON extraction.
    The verifier prompt requires strict JSON only, but this keeps the node stable.
    """
    text = (text or "").strip()
    if not text:
        return {}

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    return {}


def _normalize_issue_list(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    result: List[Dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict):
            result.append({
                "type": item.get("type"),
                "severity": item.get("severity", "medium"),
                "message": item.get("message"),
                "evidence": item.get("evidence"),
            })
        else:
            result.append({
                "type": "unknown",
                "severity": "medium",
                "message": str(item),
                "evidence": None,
            })
    return result


def _normalize_revision_advice(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


# ============================================================
# prompt builder
# ============================================================

def build_verifier_system_prompt() -> str:
    """
    Single-pass, beacon-centered, structured verifier.
    """
    return (
        "You are a strict Beacon-based code verifier.\n"
        "You do NOT generate code.\n"
        "You do NOT rewrite code.\n"
        "You perform exactly one reflection pass only.\n"
        "Your job is to compare the original beacon, rebuilt beacon, generated code, and task target.\n"
        "You must judge structural consistency, not style.\n"
        "You must check exactly these dimensions:\n"
        "1. required coverage\n"
        "2. argument binding\n"
        "3. dependency path consistency\n"
        "4. return contract consistency\n"
        "Return JSON only.\n"
        "Do not output markdown fences.\n"
        "Do not output explanations outside JSON."
    )


def build_verifier_user_prompt(
    *,
    task: Any,
    original_beacon: Any,
    rebuilt_beacon: Any,
    generated_code: str,
) -> str:
    """
    Beacon-centered verification prompt.
    """
    task_context = _compact_task_context(task)

    schema = {
        "accepted": True,
        "issues": [
            {
                "type": "required_coverage | argument_binding | dependency_path_consistency | return_contract_consistency | other",
                "severity": "high | medium | low",
                "message": "short issue description",
                "evidence": "brief evidence from beacon/code/task"
            }
        ],
        "revision_advice": [
            "short actionable advice without writing code"
        ]
    }

    parts = [
        "Perform one strict verification pass on the generated code using Beacon information.",
        "",
        "VERIFICATION TARGET:",
        "Decide whether the generated code is structurally acceptable for the task.",
        "",
        "MANDATORY CHECKS:",
        "1. required coverage: whether all task-required semantic actions are covered",
        "2. argument binding: whether parameter use and call binding align with beacon/task",
        "3. dependency path consistency: whether called symbols / dependency path align with beacon",
        "4. return contract consistency: whether return behavior matches signature/docstring/beacon target",
        "",
        "DECISION RULE:",
        "- accepted = true only if there is no material structural inconsistency.",
        "- If any important inconsistency exists, accepted must be false.",
        "- revision_advice must be concise and actionable, but must not contain generated code.",
        "",
        "OUTPUT SCHEMA:",
        _stable_json(schema),
        "",
        "TASK_CONTEXT:",
        _stable_json(task_context),
        "",
        "ORIGINAL_BEACON:",
        _stable_json(_as_dict(original_beacon)),
        "",
        "REBUILT_BEACON:",
        _stable_json(_as_dict(rebuilt_beacon)),
        "",
        "GENERATED_CODE:",
        generated_code.strip(),
        "",
        "FINAL OUTPUT RULE:",
        "Return one JSON object only. No markdown. No extra text.",
    ]
    return "\n".join(parts).strip()


# ============================================================
# result objects
# ============================================================

@dataclass
class VerifierResult:
    accepted: bool
    issues: List[Dict[str, Any]]
    revision_advice: List[str]
    prompt_snapshot: str
    raw_response: Dict[str, Any]
    model_name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# agent
# ============================================================

class BeaconVerifierAgent:
    """
    One-pass Beacon verifier.

    Responsibilities:
    - Consume original beacon + rebuilt beacon + generated code + task
    - Run one structured reflection pass
    - Return structured verification result
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm_client = llm_client

    def run(
        self,
        *,
        task: Any,
        original_beacon: Any,
        rebuilt_beacon: Any,
        generated_code: str,
    ) -> VerifierResult:
        if not isinstance(generated_code, str) or not generated_code.strip():
            raise ValueError("generated_code must be a non-empty string.")

        system_prompt = build_verifier_system_prompt()
        user_prompt = build_verifier_user_prompt(
            task=task,
            original_beacon=original_beacon,
            rebuilt_beacon=rebuilt_beacon,
            generated_code=generated_code,
        )

        response = self.llm_client.chat(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
        )

        parsed = _extract_json_block(response.text)

        accepted = bool(parsed.get("accepted", False))
        issues = _normalize_issue_list(parsed.get("issues"))
        revision_advice = _normalize_revision_advice(parsed.get("revision_advice"))

        return VerifierResult(
            accepted=accepted,
            issues=issues,
            revision_advice=revision_advice,
            prompt_snapshot=response.prompt_snapshot,
            raw_response=response.raw_response,
            model_name=getattr(response, "model_name", None),
        )