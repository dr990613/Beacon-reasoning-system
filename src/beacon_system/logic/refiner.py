# src/beacon_system/logic/refiner.py
# -*- coding: utf-8 -*-

"""
Beacon Tree Refiner (restricted LLM-assisted formatting and audit).

Design goals:
- refine Beacon Tree formatting in a strictly bounded way
- allow only:
    1) broken-chain checking
    2) formatting normalization
    3) light explanatory notes
    4) prompt-contract enforcement
- NEVER allow:
    - changing structural facts
    - inventing dependencies
    - adding/removing functions
    - rewriting code semantics

Expected usage:
    refined = refine_beacon_tree(
        beacon_tree=beacon_tree,
        signature_hints=signature_hints,
        llm_client=llm_client,
        run_config=run_config,
    )

Notes:
- This module keeps prompt templates and interaction contract internally.
- Model credentials / provider config should be prepared outside (e.g. engine / llm layer).
- If the model output violates the contract, fallback to deterministic passthrough.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import re
from typing import Any, Dict, List, Optional


# ============================================================
# Local contracts
# ============================================================

@dataclass
class RefinerResult:
    accepted: bool
    refined_text: str
    notes: List[str]
    warnings: List[str]
    audit: Dict[str, Any]
    debug: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ============================================================
# Public API
# ============================================================

def refine_beacon_tree(
    beacon_tree: Any,
    signature_hints: Optional[Any] = None,
    llm_client: Optional[Any] = None,
    run_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Refine Beacon Tree with a restricted audit/formatting step.

    Behavior:
    - if llm_client is not provided, deterministic normalization only
    - if llm_client exists, call restricted prompt
    - if model output violates contract, fallback to deterministic normalization
    """
    tree = _as_dict(beacon_tree)
    sig = _as_dict(signature_hints) if signature_hints is not None else {}
    cfg = run_config or {}

    original_text = str(tree.get("rendered_text", "") or "").strip()
    if not original_text:
        return RefinerResult(
            accepted=False,
            refined_text="",
            notes=[],
            warnings=["empty beacon tree text"],
            audit={"mode": "empty"},
            debug={},
        ).to_dict()

    normalized_text = _deterministic_normalize_tree_text(original_text)

    # deterministic only mode
    if llm_client is None or not _refiner_enabled(cfg):
        return RefinerResult(
            accepted=True,
            refined_text=normalized_text,
            notes=[],
            warnings=[],
            audit={
                "mode": "deterministic_only",
                "structure_changed": False,
            },
            debug={},
        ).to_dict()

    prompt = _build_refiner_prompt(
        beacon_tree_text=normalized_text,
        signature_hints=sig,
        run_config=cfg,
    )

    raw_response = None
    parsed = None
    warnings: List[str] = []

    try:
        raw_response = _call_llm_refiner(
            llm_client=llm_client,
            prompt=prompt,
            run_config=cfg,
        )
        parsed = _parse_refiner_response(raw_response)
    except Exception as exc:
        warnings.append(f"refiner llm call failed: {exc}")
        return RefinerResult(
            accepted=True,
            refined_text=normalized_text,
            notes=[],
            warnings=warnings,
            audit={
                "mode": "fallback_after_llm_failure",
                "structure_changed": False,
            },
            debug={"raw_response": raw_response},
        ).to_dict()

    if parsed is None:
        warnings.append("refiner output parse failed; used deterministic fallback")
        return RefinerResult(
            accepted=True,
            refined_text=normalized_text,
            notes=[],
            warnings=warnings,
            audit={
                "mode": "fallback_after_parse_failure",
                "structure_changed": False,
            },
            debug={"raw_response": raw_response},
        ).to_dict()

    candidate_text = str(parsed.get("refined_text", "") or "").strip()
    notes = list(parsed.get("notes", []) or [])
    model_audit = parsed.get("audit", {}) or {}

    candidate_text = _deterministic_normalize_tree_text(candidate_text)

    valid, audit = _validate_refined_tree(
        original_text=normalized_text,
        candidate_text=candidate_text,
    )

    merged_audit = {
        "mode": "llm_refiner",
        "model_audit": model_audit,
        **audit,
    }

    if not valid:
        warnings.append("refiner output violated structural contract; used deterministic fallback")
        return RefinerResult(
            accepted=True,
            refined_text=normalized_text,
            notes=[],
            warnings=warnings,
            audit=merged_audit,
            debug={"raw_response": raw_response},
        ).to_dict()

    return RefinerResult(
        accepted=True,
        refined_text=candidate_text,
        notes=notes[:8],
        warnings=warnings,
        audit=merged_audit,
        debug={"raw_response": raw_response},
    ).to_dict()


# ============================================================
# Prompt construction
# ============================================================

def _build_refiner_prompt(
    beacon_tree_text: str,
    signature_hints: Dict[str, Any],
    run_config: Dict[str, Any],
) -> str:
    max_notes = int(run_config.get("logic", {}).get("refiner_max_notes", 5))
    sig_summary = _compact_signature_summary(signature_hints)

    beacon_rules = """
Beacon definition:
A Beacon Tree is a compact program-semantic backbone, not a full AST dump and not a generic call graph.
It is built to preserve the main semantic chain that explains how the program produces observable outputs.

Beacon Local Logic:
1. Observable output nodes are the starting anchors, such as return / yield / print / log / file-write.
2. Local Beacon includes the backward dependency closure of those output anchors.
3. Pure validation / defensive guard logic may be excluded when it does not belong to the main functional semantics.
4. The final local result is reduced and normalized for readability.

Beacon Global Logic:
1. Local beacons are lifted to function-level global seeds.
2. If a beacon-relevant call site appears in a function's semantic chain, the callee beacon may be propagated into the caller.
3. If a callee return flow affects a beacon-relevant statement in the caller, the callee beacon may be propagated into the caller.
4. Global state may be propagated conservatively when relevant reads/writes participate in the beacon chain.
5. Program Beacon is organized around entry functions, usually main or public API-like entry points.

Refiner authority limits:
You are NOT allowed to rebuild Beacon logic.
You are NOT allowed to add or remove functions, statements, dependencies, or call relations.
You are NOT allowed to change line numbers or rewrite statement code.
You may only:
- detect obvious visible formatting issues,
- detect obvious visible chain breaks,
- normalize formatting,
- add a few short audit notes.
- Each function should at least retain its dependencies, function name, and basic parameters, for example, "def xxx(x=y)".
""".strip()

    return f"""
You are a restricted Beacon Tree auditor and formatter.

You must understand Beacon using the rules below, but you must NOT recompute or rewrite Beacon structure.

{beacon_rules}

Your job is ONLY to:
1. check whether the provided Beacon Tree appears visibly broken or structurally inconsistent,
2. normalize formatting,
3. provide at most {max_notes} short audit notes,
4. return output in the required JSON contract.

You MUST NOT:
- add new functions,
- remove existing functions,
- change function names,
- change line numbers,
- change statement code text,
- invent new dependencies,
- invent call propagation,
- invent return-flow propagation,
- rewrite the semantic content of the tree.

Required JSON output:
{{
  "refined_text": "<the normalized Beacon Tree text only>",
  "notes": ["short note 1", "short note 2"],
  "audit": {{
    "broken_chain_suspected": true,
    "formatting_fixed": true,
    "semantic_change_attempted": false
  }}
}}

Additional function signature and binding context:
{sig_summary}

Beacon Tree to audit:
<<<BEACON_TREE
{beacon_tree_text}
BEACON_TREE>>>

Return JSON only.
""".strip()
def _compact_signature_summary(signature_hints: Dict[str, Any]) -> str:
    if not signature_hints:
        return "None"

    functions = signature_hints.get("functions", []) or []
    if not functions:
        return "None"

    lines: List[str] = []
    for fn in functions[:20]:
        fn_name = str(fn.get("function_name", "unknown"))
        signature = str(fn.get("signature", "") or "")
        params = fn.get("parameters", []) or []
        param_names = [str(p.get("name", "")) for p in params if p.get("name")]
        lines.append(f"- {fn_name}: {signature or 'unknown signature'} | params={param_names}")
    return "\n".join(lines)


# ============================================================
# LLM interaction
# ============================================================

def _call_llm_refiner(llm_client: Any, prompt: str, run_config: Dict[str, Any]) -> str:
    """
    Compatible with a simple injected llm_client.

    Expected client styles supported:
    1) llm_client.complete(prompt=...)
    2) llm_client.chat(messages=[...])
    """
    temperature = float(run_config.get("logic", {}).get("refiner_temperature", 0.0))
    max_tokens = int(run_config.get("logic", {}).get("refiner_max_tokens", 1200))

    if hasattr(llm_client, "complete"):
        resp = llm_client.complete(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _extract_text_from_llm_response(resp)

    if hasattr(llm_client, "chat"):
        resp = llm_client.chat(
            messages=[
                {"role": "system", "content": "You are a strict JSON-only formatter."},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return _extract_text_from_llm_response(resp)

    raise TypeError("unsupported llm_client interface for refiner")


def _extract_text_from_llm_response(resp: Any) -> str:
    if resp is None:
        return ""
    if isinstance(resp, str):
        return resp
    if isinstance(resp, dict):
        # common patterns
        if "text" in resp:
            return str(resp["text"])
        if "content" in resp:
            return str(resp["content"])
        if "message" in resp and isinstance(resp["message"], dict):
            return str(resp["message"].get("content", ""))
        if "choices" in resp and resp["choices"]:
            first = resp["choices"][0]
            if isinstance(first, dict):
                if "text" in first:
                    return str(first["text"])
                if "message" in first and isinstance(first["message"], dict):
                    return str(first["message"].get("content", ""))
    return str(resp)


# ============================================================
# Parsing and validation
# ============================================================

def _parse_refiner_response(raw_response: str) -> Optional[Dict[str, Any]]:
    text = str(raw_response or "").strip()
    if not text:
        return None

    # pure json
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # fenced json
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    # first object fallback
    m2 = re.search(r"(\{.*\})", text, flags=re.DOTALL)
    if m2:
        try:
            obj = json.loads(m2.group(1))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def _validate_refined_tree(original_text: str, candidate_text: str) -> tuple[bool, Dict[str, Any]]:
    """
    Strictly block semantic rewriting.

    Allowed:
    - whitespace normalization
    - indentation normalization
    - duplicate blank line cleanup

    Not allowed:
    - changed line numbers
    - changed function names
    - changed statement code text
    - changed count of function headers
    """
    orig = _tree_facts(original_text)
    cand = _tree_facts(candidate_text)

    audit = {
        "original_function_headers": orig["function_headers"],
        "candidate_function_headers": cand["function_headers"],
        "original_statement_keys": orig["statement_keys"],
        "candidate_statement_keys": cand["statement_keys"],
        "structure_changed": False,
    }

    if orig["function_headers"] != cand["function_headers"]:
        audit["structure_changed"] = True
        audit["reason"] = "function headers changed"
        return False, audit

    if orig["statement_keys"] != cand["statement_keys"]:
        audit["structure_changed"] = True
        audit["reason"] = "statement set changed"
        return False, audit

    return True, audit


def _tree_facts(text: str) -> Dict[str, Any]:
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    function_headers: List[str] = []
    statement_keys: List[str] = []

    for ln in lines:
        s = ln.strip()
        if s.startswith("Function ") or s.startswith("├─ Function ") or s.startswith("└─ Function "):
            function_headers.append(_normalize_spaces(s))
            continue

        m = re.search(r"\[([^\]]+)\]\s+line\s+(\d+):\s+(.*)$", s)
        if m:
            fn_name = m.group(1).strip()
            line_no = m.group(2).strip()
            code = _normalize_spaces(m.group(3).strip())
            statement_keys.append(f"{fn_name}|{line_no}|{code}")
            continue

        m2 = re.search(r"\[visited\]\s+\[([^\]]+)\]\s+line\s+(\d+):\s+(.*)$", s)
        if m2:
            fn_name = m2.group(1).strip()
            line_no = m2.group(2).strip()
            code = _normalize_spaces(m2.group(3).strip())
            statement_keys.append(f"visited|{fn_name}|{line_no}|{code}")

    return {
        "function_headers": function_headers,
        "statement_keys": statement_keys,
    }


# ============================================================
# Deterministic normalization
# ============================================================

def _deterministic_normalize_tree_text(text: str) -> str:
    lines = text.splitlines()

    normalized: List[str] = []
    prev_blank = False

    for raw in lines:
        line = raw.rstrip()

        # collapse repeated blank lines
        if not line.strip():
            if prev_blank:
                continue
            prev_blank = True
            normalized.append("")
            continue

        prev_blank = False

        # normalize tabs
        line = line.replace("\t", "    ")

        # keep tree glyphs and content, only trim trailing spaces
        normalized.append(line)

    return "\n".join(normalized).strip()


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# Config helpers
# ============================================================

def _refiner_enabled(run_config: Dict[str, Any]) -> bool:
    logic_cfg = run_config.get("logic", {}) if isinstance(run_config, dict) else {}
    return bool(logic_cfg.get("enable_refiner_llm", False))


def _as_dict(obj: Any) -> Dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)
    return {"value": obj}