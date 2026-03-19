# src/test_model_code_generation_probe.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "").strip()
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "").strip()

OUTPUT_DIR = Path(os.getenv("OUTPUTS_DIR", "outputs/model_probe"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def require_env() -> None:
    missing = []
    if not MODEL_BASE_URL:
        missing.append("MODEL_BASE_URL")
    if not MODEL_API_KEY:
        missing.append("MODEL_API_KEY")
    if not MODEL_NAME:
        missing.append("MODEL_NAME")

    if missing:
        raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


def make_client() -> OpenAI:
    return OpenAI(
        base_url=MODEL_BASE_URL,
        api_key=MODEL_API_KEY,
    )


def strip_code_fences(text: str) -> str:
    s = (text or "").strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def find_target_function(text: str, fn_name: str) -> Optional[str]:
    s = strip_code_fences(text)
    patterns = [
        rf"(?m)^def\s+{re.escape(fn_name)}\s*\(",
        rf"(?m)^async\s+def\s+{re.escape(fn_name)}\s*\(",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if m:
            return s[m.start():].lstrip()
    return None


def parse_ok(code: str) -> tuple[bool, Optional[str]]:
    try:
        ast.parse(code)
        return True, None
    except SyntaxError as e:
        return False, f"SyntaxError: {e.msg} at line {e.lineno}, col {e.offset}"
    except Exception as e:
        return False, repr(e)


def contains_top_level_function(code: str, fn_name: str) -> bool:
    try:
        mod = ast.parse(code)
    except Exception:
        return False

    return any(
        isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == fn_name
        for n in mod.body
    )


def looks_like_explanation(text: str) -> bool:
    s = (text or "").strip().lower()
    markers = [
        "here is",
        "here's",
        "i'll",
        "i will",
        "let me",
        "explanation",
        "the function",
        "this implementation",
        "we need to",
        "okay,",
        "sure,",
    ]
    return any(m in s[:300] for m in markers)


@dataclass
class ProbeCase:
    name: str
    entry_function: str
    prompt: str


def build_cases() -> List[ProbeCase]:
    return [
        ProbeCase(
            name="simple_add",
            entry_function="add",
            prompt=(
                "Return only valid Python code.\n"
                "Do not include markdown fences.\n"
                "Do not include explanation.\n"
                "Your response must begin with: def add(\n\n"
                "Implement exactly this function:\n"
                "def add(a: int, b: int) -> int:\n"
                '    """Return a + b."""\n'
            ),
        ),
        ProbeCase(
            name="hydrate_time_style",
            entry_function="hydrate_time",
            prompt=(
                "Return only valid Python code.\n"
                "Do not include markdown fences.\n"
                "Do not include explanation text.\n"
                "Your response must begin with: def hydrate_time(\n\n"
                "Implement exactly this function:\n"
                "def hydrate_time(seconds: int, nanoseconds: int):\n"
                '    """Convert input values into a time-like object."""\n'
                "    pass\n"
            ),
        ),
        ProbeCase(
            name="strict_code_only",
            entry_function="solve",
            prompt=(
                "Write Python code only.\n"
                "No prose. No markdown. No notes. No alternatives.\n"
                "Your response must begin with: def solve(\n\n"
                "Implement:\n"
                "def solve(nums: list[int]) -> int:\n"
                "    pass\n"
            ),
        ),
    ]


def call_model(client: OpenAI, prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are a careful Python code generator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        top_p=0.95,
        max_tokens=2048,
    )
    text = resp.choices[0].message.content or ""
    return text


def run_case(client: OpenAI, case: ProbeCase, case_dir: Path) -> Dict[str, Any]:
    case_dir.mkdir(parents=True, exist_ok=True)

    raw = call_model(client, case.prompt)
    (case_dir / "prompt.txt").write_text(case.prompt, encoding="utf-8")
    (case_dir / "raw_output.txt").write_text(raw, encoding="utf-8")

    stripped = strip_code_fences(raw)
    (case_dir / "stripped_output.txt").write_text(stripped, encoding="utf-8")

    extracted = find_target_function(raw, case.entry_function)
    if extracted is not None:
        (case_dir / "extracted_target.py").write_text(extracted, encoding="utf-8")

    raw_parse_ok, raw_parse_err = parse_ok(stripped)
    extracted_parse_ok = False
    extracted_parse_err = None
    extracted_has_target = False

    if extracted is not None:
        extracted_parse_ok, extracted_parse_err = parse_ok(extracted)
        extracted_has_target = contains_top_level_function(extracted, case.entry_function)

    result = {
        "case": case.name,
        "entry_function": case.entry_function,
        "raw_starts_with_def": stripped.lstrip().startswith(f"def {case.entry_function}(")
        or stripped.lstrip().startswith(f"async def {case.entry_function}("),
        "target_function_found_by_regex": extracted is not None,
        "raw_parse_ok": raw_parse_ok,
        "raw_parse_err": raw_parse_err,
        "extracted_parse_ok": extracted_parse_ok,
        "extracted_parse_err": extracted_parse_err,
        "extracted_has_target": extracted_has_target,
        "looks_like_explanation": looks_like_explanation(raw),
        "raw_preview": raw[:500],
    }

    (case_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> int:
    require_env()
    client = make_client()

    run_id = OUTPUT_DIR / "latest"
    run_id.mkdir(parents=True, exist_ok=True)

    summary: List[Dict[str, Any]] = []
    failures = 0

    print("=" * 100)
    print("Model code generation probe")
    print(f"base_url   : {MODEL_BASE_URL}")
    print(f"model_name : {MODEL_NAME}")
    print(f"output_dir : {run_id}")
    print("=" * 100)

    for case in build_cases():
        case_dir = run_id / case.name
        print(f"[RUN] {case.name}")
        try:
            result = run_case(client, case, case_dir)
            summary.append(result)

            ok = (
                result["target_function_found_by_regex"]
                and result["extracted_parse_ok"]
                and result["extracted_has_target"]
            )

            if ok:
                print(f"[OK]  {case.name}")
            else:
                failures += 1
                print(f"[FAIL] {case.name}")
                print(json.dumps(result, ensure_ascii=False, indent=2))
        except Exception as e:
            failures += 1
            err = {
                "case": case.name,
                "error": repr(e),
            }
            summary.append(err)
            (case_dir / "error.json").write_text(
                json.dumps(err, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[FAIL] {case.name}: {e}")

    summary_path = run_id / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("-" * 100)
    print(f"Summary written to: {summary_path}")
    print(f"Failures: {failures}")

    if failures == 0:
        print("[PASS] Model can return parseable target-function code for all probe cases.")
        return 0

    print("[FAIL] At least one probe case did not produce target-function code.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())