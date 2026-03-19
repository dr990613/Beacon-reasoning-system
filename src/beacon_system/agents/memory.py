# src/beacon_system/agents/memory.py
# -*- coding: utf-8 -*-

"""
Experience memory for Beacon agent workflow.

Scope:
- lightweight local JSONL-backed memory
- read relevant past experience for a task
- write current run experience after generation / verification / execution
- produce transparent, debuggable memory artifacts

Non-goals:
- no vector DB
- no embedding retrieval
- no logic integration
- no complex long-term memory policy
"""

from __future__ import annotations

import json
import os
import re
import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import (
    Constraints,
    ExecResult,
    MemoryReadResult,
    MemoryRecord,
    MemoryWriteResult,
    TaskObject,
    VerifierResult,
)


def _safe_text(text: Optional[str]) -> str:
    return str(text or "").strip()


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def _truncate(text: str, max_chars: int = 300) -> str:
    text = _safe_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + " ..."


def _tokenize(text: str) -> List[str]:
    """
    Very lightweight tokenizer for retrieval scoring.
    """
    text = _safe_text(text).lower()
    if not text:
        return []
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_./:-]{1,}", text)
    # keep order while deduplicating
    seen = set()
    result: List[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result


def _json_dumps(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)


def _make_record_id(task: TaskObject, salt: str = "") -> str:
    raw = f"{task.id}|{task.target.get('file')}|{task.target.get('qualname')}|{salt}|{time.time_ns()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _build_query_text(task: TaskObject, constraints: Optional[Constraints]) -> str:
    parts = [
        task.id,
        task.lang,
        task.level,
        task.target.get("file", ""),
        task.target.get("qualname", ""),
        task.spec,
    ]

    if constraints is not None:
        parts.extend(list(constraints.required_symbols or ()))
        parts.extend(list(constraints.required_calls or ()))

    return "\n".join([_safe_text(x) for x in parts if _safe_text(x)])


def _build_tags(
    *,
    task: TaskObject,
    constraints: Optional[Constraints],
    extra_tags: Optional[Sequence[str]] = None,
) -> Tuple[str, ...]:
    tags: List[str] = []

    lang = _safe_text(task.lang)
    if lang:
        tags.append(f"lang:{lang}")

    level = _safe_text(task.level)
    if level:
        tags.append(f"level:{level}")

    target_file = _safe_text(task.target.get("file"))
    if target_file:
        tags.append(f"file:{target_file}")

    target_qualname = _safe_text(task.target.get("qualname"))
    if target_qualname:
        tags.append(f"qualname:{target_qualname}")

    if constraints is not None:
        for sym in list(constraints.required_symbols or ())[:8]:
            sym = _safe_text(sym)
            if sym:
                tags.append(f"symbol:{sym}")
        for call in list(constraints.required_calls or ())[:8]:
            call = _safe_text(call)
            if call:
                tags.append(f"call:{call}")

    if extra_tags:
        for tag in extra_tags:
            tag = _safe_text(tag)
            if tag:
                tags.append(tag)

    # stable dedupe
    seen = set()
    result: List[str] = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            result.append(t)

    return tuple(result)


def _score_record(
    *,
    query_tokens: Sequence[str],
    task: TaskObject,
    constraints: Optional[Constraints],
    record: MemoryRecord,
) -> float:
    """
    Deterministic lightweight retrieval score.
    """
    score = 0.0

    target_file = _safe_text(task.target.get("file"))
    target_qualname = _safe_text(task.target.get("qualname"))

    record_key = _safe_text(record.key).lower()
    record_blob = _json_dumps(record.value).lower()
    record_tags = " ".join(record.tags or ()).lower()

    # Strong target matches
    if target_qualname and target_qualname.lower() in record_key:
        score += 4.0
    if target_qualname and target_qualname.lower() in record_blob:
        score += 3.0

    if target_file and target_file.lower() in record_key:
        score += 2.0
    if target_file and target_file.lower() in record_blob:
        score += 1.5

    # Query-token overlap
    token_hits = 0
    for token in query_tokens:
        if token in record_key or token in record_blob or token in record_tags:
            token_hits += 1
    score += min(5.0, token_hits * 0.35)

    # Constraint hints
    if constraints is not None:
        for sym in constraints.required_symbols or ():
            sym = _safe_text(sym).lower()
            if sym and (sym in record_blob or f"symbol:{sym}" in record_tags):
                score += 0.8

        for call in constraints.required_calls or ():
            call = _safe_text(call).lower()
            if call and (call in record_blob or f"call:{call}" in record_tags):
                score += 1.0

    # Prefer successful past experience slightly
    status = _safe_text(record.value.get("status")).lower()
    if status == "success":
        score += 1.0
    elif status == "partial":
        score += 0.2
    elif status == "failure":
        score -= 0.5

    return round(score, 4)


@dataclass
class ExperienceMemory:
    """
    Minimal JSONL-backed experience memory.

    Storage format:
    - one MemoryRecord-like JSON object per line

    Read path:
    - load all records
    - score by simple overlap heuristics
    - return top-k records

    Write path:
    - append one normalized record
    """
    store_path: str = "outputs/memory/experience.jsonl"
    max_read_items: int = 5
    print_io: bool = False

    def _print(self, message: str) -> None:
        if self.print_io:
            print(f"[ExperienceMemory] {message}")

    def _load_records(self) -> List[MemoryRecord]:
        path = os.path.abspath(self.store_path)
        if not os.path.isfile(path):
            return []

        records: List[MemoryRecord] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    raw = json.loads(line)
                    records.append(
                        MemoryRecord(
                            key=_safe_text(raw.get("key")),
                            value=dict(raw.get("value") or {}),
                            source_run_id=_safe_text(raw.get("source_run_id")),
                            tags=tuple(raw.get("tags") or ()),
                            meta=dict(raw.get("meta") or {}),
                        )
                    )
                except Exception:
                    # skip bad lines instead of failing the whole read
                    continue
        return records

    def _append_record(self, record: MemoryRecord) -> None:
        _ensure_parent_dir(self.store_path)
        with open(self.store_path, "a", encoding="utf-8") as f:
            f.write(_json_dumps({
                "key": record.key,
                "value": record.value,
                "source_run_id": record.source_run_id,
                "tags": list(record.tags),
                "meta": record.meta,
            }) + "\n")

    def read(
        self,
        *,
        task: TaskObject,
        constraints: Optional[Constraints] = None,
        top_k: Optional[int] = None,
    ) -> MemoryReadResult:
        self._print(f"read start: task={task.id}")

        records = self._load_records()
        if not records:
            self._print("read done: no records")
            return MemoryReadResult(
                items=(),
                meta={
                    "memory": "ExperienceMemory",
                    "store_path": os.path.abspath(self.store_path),
                    "loaded_count": 0,
                    "returned_count": 0,
                },
            )

        query_text = _build_query_text(task, constraints)
        query_tokens = _tokenize(query_text)

        scored: List[Tuple[float, MemoryRecord]] = []
        for record in records:
            score = _score_record(
                query_tokens=query_tokens,
                task=task,
                constraints=constraints,
                record=record,
            )
            if score > 0:
                scored.append((score, record))

        scored.sort(
            key=lambda x: (
                -x[0],
                x[1].source_run_id,
                x[1].key,
            )
        )

        limit = int(top_k if top_k is not None else self.max_read_items)
        limit = max(1, limit)

        selected_records = tuple(record for _, record in scored[:limit])

        self._print(f"read done: selected={len(selected_records)}")
        return MemoryReadResult(
            items=selected_records,
            meta={
                "memory": "ExperienceMemory",
                "store_path": os.path.abspath(self.store_path),
                "loaded_count": len(records),
                "matched_count": len(scored),
                "returned_count": len(selected_records),
                "query_tokens": query_tokens[:30],
            },
        )

    def format_for_prompt(
        self,
        read_result: MemoryReadResult,
        *,
        max_items: int = 5,
    ) -> str:
        """
        Render retrieved memory into compact text for prompt injection.
        """
        items = list(read_result.items or ())[:max(1, int(max_items))]
        if not items:
            return ""

        chunks: List[str] = []
        for idx, rec in enumerate(items, start=1):
            value = dict(rec.value or {})
            lines = [
                f"Memory {idx}:",
                f"- key: {rec.key}",
                f"- status: {_safe_text(value.get('status')) or 'unknown'}",
                f"- target: {_safe_text(value.get('target_file'))}::{_safe_text(value.get('target_qualname'))}",
                f"- summary: {_truncate(_safe_text(value.get('summary')), 240)}",
            ]

            used_symbols = value.get("used_required_symbols") or []
            used_calls = value.get("used_required_calls") or []
            if used_symbols:
                lines.append(f"- used_required_symbols: {', '.join(list(used_symbols)[:8])}")
            if used_calls:
                lines.append(f"- used_required_calls: {', '.join(list(used_calls)[:8])}")

            verifier_ok = value.get("verifier_ok")
            if verifier_ok is not None:
                lines.append(f"- verifier_ok: {verifier_ok}")

            exec_status = _safe_text(value.get("exec_status"))
            if exec_status:
                lines.append(f"- exec_status: {exec_status}")

            notes = value.get("notes") or []
            if notes:
                lines.append(f"- notes: {' | '.join([_truncate(_safe_text(x), 120) for x in list(notes)[:4]])}")

            chunks.append("\n".join(lines))

        return "\n\n".join(chunks).strip()

    def write(
        self,
        *,
        task: TaskObject,
        constraints: Optional[Constraints] = None,
        run_id: str = "",
        status: str = "success",  # success | partial | failure
        summary: str = "",
        selected_thought_id: str = "",
        verifier_result: Optional[VerifierResult] = None,
        exec_result: Optional[ExecResult] = None,
        used_required_symbols: Optional[Sequence[str]] = None,
        used_required_calls: Optional[Sequence[str]] = None,
        notes: Optional[Sequence[str]] = None,
        extra_tags: Optional[Sequence[str]] = None,
        extra_value: Optional[Dict[str, Any]] = None,
    ) -> MemoryWriteResult:
        """
        Append one normalized experience record.
        """
        self._print(f"write start: task={task.id} status={status}")

        status = _safe_text(status).lower() or "success"
        if status not in {"success", "partial", "failure"}:
            status = "partial"

        target_file = _safe_text(task.target.get("file"))
        target_qualname = _safe_text(task.target.get("qualname"))

        value: Dict[str, Any] = {
            "task_id": task.id,
            "lang": task.lang,
            "level": task.level,
            "target_file": target_file,
            "target_qualname": target_qualname,
            "spec": _truncate(task.spec, 500),
            "status": status,
            "summary": _truncate(summary, 800),
            "selected_thought_id": _safe_text(selected_thought_id),
            "required_symbols": list(constraints.required_symbols or ()) if constraints is not None else [],
            "required_calls": list(constraints.required_calls or ()) if constraints is not None else [],
            "used_required_symbols": list(used_required_symbols or ()),
            "used_required_calls": list(used_required_calls or ()),
            "notes": list(notes or ()),
        }

        if verifier_result is not None:
            value["verifier_ok"] = bool(verifier_result.ok)
            value["violation_count"] = len(verifier_result.violations or ())
            value["directive_count"] = len(verifier_result.directives or ())

        if exec_result is not None:
            value["exec_status"] = exec_result.status
            value["return_code"] = exec_result.return_code
            value["trace_preview"] = _truncate(exec_result.trace, 500)
            value["exec_metrics"] = dict(exec_result.metrics or {})

        if extra_value:
            value.update(dict(extra_value))

        tags = _build_tags(
            task=task,
            constraints=constraints,
            extra_tags=extra_tags,
        )

        record = MemoryRecord(
            key=f"{target_file}::{target_qualname}",
            value=value,
            source_run_id=_safe_text(run_id),
            tags=tags,
            meta={
                "memory": "ExperienceMemory",
                "record_id": _make_record_id(task, salt=run_id),
                "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        )

        self._append_record(record)

        self._print("write done: 1 record appended")
        return MemoryWriteResult(
            written=(record,),
            skipped=(),
            meta={
                "memory": "ExperienceMemory",
                "store_path": os.path.abspath(self.store_path),
                "written_count": 1,
            },
        )


def read_memory(
    *,
    task: TaskObject,
    constraints: Optional[Constraints] = None,
    store_path: str = "outputs/memory/experience.jsonl",
    top_k: int = 5,
    print_io: bool = False,
) -> MemoryReadResult:
    """
    Convenience function for direct memory read.
    """
    mem = ExperienceMemory(
        store_path=store_path,
        max_read_items=top_k,
        print_io=print_io,
    )
    return mem.read(task=task, constraints=constraints, top_k=top_k)


def write_memory(
    *,
    task: TaskObject,
    constraints: Optional[Constraints] = None,
    run_id: str = "",
    status: str = "success",
    summary: str = "",
    selected_thought_id: str = "",
    verifier_result: Optional[VerifierResult] = None,
    exec_result: Optional[ExecResult] = None,
    used_required_symbols: Optional[Sequence[str]] = None,
    used_required_calls: Optional[Sequence[str]] = None,
    notes: Optional[Sequence[str]] = None,
    extra_tags: Optional[Sequence[str]] = None,
    extra_value: Optional[Dict[str, Any]] = None,
    store_path: str = "outputs/memory/experience.jsonl",
    print_io: bool = False,
) -> MemoryWriteResult:
    """
    Convenience function for direct memory write.
    """
    mem = ExperienceMemory(
        store_path=store_path,
        print_io=print_io,
    )
    return mem.write(
        task=task,
        constraints=constraints,
        run_id=run_id,
        status=status,
        summary=summary,
        selected_thought_id=selected_thought_id,
        verifier_result=verifier_result,
        exec_result=exec_result,
        used_required_symbols=used_required_symbols,
        used_required_calls=used_required_calls,
        notes=notes,
        extra_tags=extra_tags,
        extra_value=extra_value,
    )