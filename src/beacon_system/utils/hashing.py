from __future__ import annotations

import hashlib


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()
