from __future__ import annotations

from pathlib import Path


def replace_function(file_path: str, function_name: str, replacement_source: str) -> None:
    path = Path(file_path)
    src = path.read_text()
    marker = f"def {function_name}("
    start = src.find(marker)
    if start < 0:
        raise ValueError(f"function {function_name} not found")
    line_start = src.rfind("\n", 0, start) + 1
    end = src.find("\ndef ", start + 1)
    if end < 0:
        end = len(src)
    new_src = src[:line_start] + replacement_source.rstrip() + "\n" + src[end:]
    path.write_text(new_src)
