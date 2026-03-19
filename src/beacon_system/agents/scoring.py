# src/beacon_system/agents/scoring.py
# -*- coding: utf-8 -*-

"""
Lightweight thought scoring and pruning.

Scope:
- Score ThoughtCandidate objects using simple deterministic heuristics.
- Keep the scoring explainable and stable.
- Select top-k thoughts for downstream generation.

Non-goals:
- no model calls
- no execution
- no verifier logic duplication
- no complex reranking framework
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from ..types import Constraints, TaskObject, ThoughtCandidate, ThoughtScore


def _safe_text(text: Any) -> str:
    return str(text or "").strip()


def _normalize_text(text: Any) -> str:
    return _safe_text(text).lower()


def _contains_word(text: str, token: str) -> bool:
    token = str(token or "").strip()
    if not token:
        return False
    pattern = r"\b" + re.escape(token.lower()) + r"\b"
    return re.search(pattern, text.lower()) is not None


def _contains_phrase(text: str, token: str) -> bool:
    token = _normalize_text(token)
    if not token:
        return False
    return token in _normalize_text(text)


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


def _thought_full_text(thought: ThoughtCandidate) -> str:
    parts: List[str] = [thought.text, thought.rationale]
    parts.extend(list(thought.steps or ()))
    parts.extend(list(thought.assumptions or ()))
    return "\n".join([_safe_text(x) for x in parts if _safe_text(x)])


def _count_hits(text: str, items: Iterable[str]) -> int:
    hits = 0
    for item in items:
        token = str(item or "").strip()
        if not token:
            continue
        if _contains_phrase(text, token) or _contains_word(text, token.split(".")[-1]):
            hits += 1
    return hits


def _simplicity_penalty(thought: ThoughtCandidate) -> float:
    steps_n = len(thought.steps or ())
    assumptions_n = len(thought.assumptions or ())
    text_len = len(_thought_full_text(thought))

    penalty = 0.0

    if steps_n > 6:
        penalty += 1.0 + 0.2 * (steps_n - 6)
    if assumptions_n > 4:
        penalty += 0.5 + 0.2 * (assumptions_n - 4)
    if text_len > 1200:
        penalty += min(2.0, (text_len - 1200) / 600.0)

    return penalty


def _forbidden_risk_penalty(text: str, constraints: Constraints) -> Tuple[float, List[str]]:
    penalty = 0.0
    reasons: List[str] = []

    forbidden_specs = _as_seq(_get_attr(constraints, "forbidden_specs", ()))

    for spec in forbidden_specs:
        if isinstance(spec, str):
            token = spec.strip()
            if token and _contains_phrase(text, token):
                penalty += 2.0
                reasons.append(f"mentions forbidden text: {token!r}")
        elif isinstance(spec, dict):
            token = str(spec.get("value") or spec.get("pattern") or spec.get("contains") or "").strip()
            if token and _contains_phrase(text, token):
                penalty += 2.0
                reasons.append(f"mentions forbidden spec token: {token!r}")

    return penalty, reasons


def _contract_fit_score(text: str, task: TaskObject) -> Tuple[float, List[str]]:
    score = 0.0
    reasons: List[str] = []

    target_file = str(task.target.get("file") or "").strip()
    target_qualname = str(task.target.get("qualname") or "").strip()
    spec = str(task.spec or "").strip()

    if target_qualname and (_contains_phrase(text, target_qualname) or _contains_word(text, target_qualname.split(".")[-1])):
        score += 2.0
        reasons.append("references target qualname")

    if target_file:
        base = target_file.split("/")[-1]
        if base and _contains_phrase(text, base):
            score += 1.0
            reasons.append("references target file")

    if spec:
        spec_words = [w for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", spec)[:12]]
        spec_hits = sum(1 for w in spec_words if _contains_word(text, w))
        if spec_hits > 0:
            add = min(2.0, 0.25 * spec_hits)
            score += add
            reasons.append(f"matches task spec keywords: {spec_hits}")

    return score, reasons


def _beacon_coverage_score(text: str, ir: Any, constraints: Constraints) -> Tuple[float, Dict[str, int], List[str]]:
    reasons: List[str] = []

    required_symbols = tuple(_get_attr(constraints, "required_symbols", ()) or ())
    required_calls = tuple(_get_attr(constraints, "required_calls", ()) or ())

    symbol_hits = _count_hits(text, required_symbols)
    call_hits = _count_hits(text, required_calls)

    score = 0.0
    if required_symbols:
        score += 1.5 * (symbol_hits / max(1, len(required_symbols)))
    if required_calls:
        score += 2.0 * (call_hits / max(1, len(required_calls)))

    if symbol_hits > 0:
        reasons.append(f"covers required symbols: {symbol_hits}/{len(required_symbols)}")
    if call_hits > 0:
        reasons.append(f"covers required calls: {call_hits}/{len(required_calls)}")

    entry = _as_dict(_get_attr(ir, "entry", {}))
    entry_qualname = str(entry.get("qualname") or "").strip()
    if entry_qualname and (_contains_phrase(text, entry_qualname) or _contains_word(text, entry_qualname.split(".")[-1])):
        score += 0.5
        reasons.append("references Beacon entry")

    node_text_hits = 0
    nodes = _as_seq(_get_attr(ir, "nodes", ()))
    for node in nodes[:8]:
        token = str(_get_attr(node, "text", "") or "").strip()
        if token and len(token) <= 80 and _contains_phrase(text, token):
            node_text_hits += 1
    if node_text_hits > 0:
        score += min(1.0, 0.25 * node_text_hits)
        reasons.append(f"references Beacon node text: {node_text_hits}")

    return score, {
        "required_symbol_hits": symbol_hits,
        "required_symbol_total": len(required_symbols),
        "required_call_hits": call_hits,
        "required_call_total": len(required_calls),
    }, reasons


@dataclass
class ThoughtScorer:
    keep_top_k: int = 1
    print_io: bool = False

    def _print(self, message: str) -> None:
        if self.print_io:
            print(f"[ThoughtScorer] {message}")

    def score_one(
        self,
        *,
        task: TaskObject,
        ir: Any,
        constraints: Constraints,
        thought: ThoughtCandidate,
    ) -> ThoughtScore:
        text = _thought_full_text(thought)

        contract_fit, contract_reasons = _contract_fit_score(text, task)
        beacon_cov, beacon_cov_meta, beacon_reasons = _beacon_coverage_score(text, ir, constraints)
        forbidden_penalty, forbidden_reasons = _forbidden_risk_penalty(text, constraints)
        simplicity_penalty = _simplicity_penalty(thought)

        total = contract_fit + beacon_cov - forbidden_penalty - simplicity_penalty

        reasons: List[str] = []
        reasons.extend(contract_reasons)
        reasons.extend(beacon_reasons)
        reasons.extend(forbidden_reasons)

        if simplicity_penalty > 0:
            reasons.append(f"simplicity penalty: {simplicity_penalty:.2f}")

        return ThoughtScore(
            thought_id=thought.id,
            total=round(total, 4),
            subscores={
                "contract_fit": round(contract_fit, 4),
                "beacon_coverage": round(beacon_cov, 4),
                "forbidden_risk_penalty": round(-forbidden_penalty, 4),
                "simplicity_penalty": round(-simplicity_penalty, 4),
                **beacon_cov_meta,
            },
            reasons=tuple(reasons),
            meta={
                "scorer": "ThoughtScorer",
                "text_len": len(text),
                "step_count": len(thought.steps or ()),
                "assumption_count": len(thought.assumptions or ()),
            },
        )

    def score_all(
        self,
        *,
        task: TaskObject,
        ir: Any,
        constraints: Constraints,
        thoughts: Sequence[ThoughtCandidate],
    ) -> Sequence[ThoughtScore]:
        self._print(f"start scoring thoughts={len(thoughts)}")

        results: List[ThoughtScore] = []
        for thought in thoughts:
            score = self.score_one(
                task=task,
                ir=ir,
                constraints=constraints,
                thought=thought,
            )
            results.append(score)

        results.sort(key=lambda x: (-x.total, x.thought_id))
        self._print(f"scoring done scores={len(results)}")
        return tuple(results)

    def select_top(
        self,
        *,
        thoughts: Sequence[ThoughtCandidate],
        scores: Sequence[ThoughtScore],
        keep_top_k: int | None = None,
    ) -> Sequence[ThoughtCandidate]:
        k = int(keep_top_k if keep_top_k is not None else self.keep_top_k)
        k = max(1, k)

        score_map: Dict[str, ThoughtScore] = {s.thought_id: s for s in scores}
        ranked = sorted(
            thoughts,
            key=lambda t: (
                -(score_map.get(t.id).total if t.id in score_map else float("-inf")),
                t.id,
            ),
        )

        selected = tuple(ranked[:k])
        self._print(f"selected top_k={len(selected)}")
        return selected


def score_thoughts(
    *,
    task: TaskObject,
    ir: Any,
    constraints: Constraints,
    thoughts: Sequence[ThoughtCandidate],
    print_io: bool = False,
) -> Sequence[ThoughtScore]:
    scorer = ThoughtScorer(print_io=print_io)
    return scorer.score_all(
        task=task,
        ir=ir,
        constraints=constraints,
        thoughts=thoughts,
    )


def select_top_thoughts(
    *,
    thoughts: Sequence[ThoughtCandidate],
    scores: Sequence[ThoughtScore],
    keep_top_k: int = 1,
    print_io: bool = False,
) -> Sequence[ThoughtCandidate]:
    scorer = ThoughtScorer(keep_top_k=keep_top_k, print_io=print_io)
    return scorer.select_top(
        thoughts=thoughts,
        scores=scores,
        keep_top_k=keep_top_k,
    )