from __future__ import annotations

from pathlib import Path


def assemble_context(file_path: str, source: str) -> dict:
    lines = source.splitlines()
    imports = [ln.strip() for ln in lines if ln.strip().startswith(("import ", "from "))]
    return {
        "file": str(Path(file_path)),
        "imports": imports,
        "source": source,
    }
