from __future__ import annotations


def make_anchor(namespace: str, local_id: str) -> str:
    return f"{namespace}::{local_id}"
