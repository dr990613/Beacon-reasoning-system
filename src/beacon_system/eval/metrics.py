from __future__ import annotations


def pass_rate(results: list[bool]) -> float:
    if not results:
        return 0.0
    return sum(1 for x in results if x) / len(results)
