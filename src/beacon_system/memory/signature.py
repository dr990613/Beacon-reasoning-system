from __future__ import annotations

import hashlib
import json


def compute_signature(payload: dict) -> str:
    stable = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(stable.encode()).hexdigest()
