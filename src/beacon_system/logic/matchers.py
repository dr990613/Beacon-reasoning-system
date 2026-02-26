from __future__ import annotations


def match_spec(spec: dict, code: str) -> bool:
    op = spec.get("op")
    value = spec.get("value", "")
    if op == "contains":
        return value in code
    if op == "not_contains":
        return value not in code
    raise ValueError(f"unknown matcher op: {op}")
