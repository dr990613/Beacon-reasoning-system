from __future__ import annotations

from enum import Enum


class ViolationType(str, Enum):
    MISSING_BEACON = "missing_beacon"
    HALLUCINATED_SYMBOL = "hallucinated_symbol"
    FORBIDDEN_PATTERN = "forbidden_pattern"
