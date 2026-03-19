# src/run_codereval_like_full_test.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import traceback
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def p(title: str, value: Any = None) -> None:
    print("\n" + "=" * 100)
    print(title)
    if value is not None:
        if isinstance(value, (dict, list)):
            print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
        else:
            print(value)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Config root must be a dict.")
    return data


def _resolve_pipeline():
    from beacon_system.pipeline import run_pipeline  # type: ignore
    return run_pipeline


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _extract_code_block(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text


def _extract_function_code(full_code: str, function_name: str) -> str:
    """
    Extract the generated target function from model output.
    Fallback to full text if direct AST extraction fails.
    """
    full_code = _extract_code_block(full_code)

    try:
        tree = ast.parse(full_code)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == function_name:
                return ast.unparse(node)
    except Exception:
        pass

    pattern = re.compile(
        rf"(^def\s+{re.escape(function_name)}\s*\(.*?(?=^\S|\Z))",
        flags=re.DOTALL | re.MULTILINE,
    )
    m = pattern.search(full_code)
    if m:
        return m.group(1).strip()

    return full_code.strip()


def _make_large_project_source() -> str:
    """
    Create a reasonably large single-file project, then we will remove one function body.
    The missing function is compute_discounted_total.
    """
    return textwrap.dedent(
        '''
        from __future__ import annotations

        from dataclasses import dataclass
        from typing import List, Dict


        @dataclass
        class OrderItem:
            name: str
            price: float
            quantity: int


        def normalize_customer_tier(tier: str) -> str:
            tier = (tier or "").strip().lower()
            if tier in {"gold", "silver", "bronze"}:
                return tier
            return "bronze"


        def compute_subtotal(items: List[OrderItem]) -> float:
            subtotal = 0.0
            for item in items:
                subtotal += item.price * item.quantity
            return round(subtotal, 2)


        def compute_discount_rate(customer_tier: str, coupon_code: str | None = None) -> float:
            tier = normalize_customer_tier(customer_tier)
            rate = 0.0

            if tier == "gold":
                rate += 0.15
            elif tier == "silver":
                rate += 0.08
            else:
                rate += 0.03

            if coupon_code:
                code = coupon_code.strip().upper()
                if code == "SAVE10":
                    rate += 0.10
                elif code == "SAVE5":
                    rate += 0.05

            # business cap
            return min(rate, 0.25)


        def compute_discounted_total(
            items: List[OrderItem],
            customer_tier: str,
            coupon_code: str | None = None,
        ) -> float:
            """
            MISSING_TARGET_FUNCTION
            The agent should restore this function.
            Expected behavior:
            1. compute subtotal using compute_subtotal(items)
            2. compute discount rate using compute_discount_rate(customer_tier, coupon_code)
            3. apply discount: subtotal * (1 - rate)
            4. return rounded float with 2 decimals
            """
            raise NotImplementedError("MISSING_TARGET_FUNCTION")


        def build_receipt(
            items: List[OrderItem],
            customer_tier: str,
            coupon_code: str | None = None,
        ) -> Dict[str, float]:
            subtotal = compute_subtotal(items)
            total = compute_discounted_total(items, customer_tier, coupon_code)
            discount = round(subtotal - total, 2)

            return {
                "subtotal": subtotal,
                "discount": discount,
                "total": total,
            }


        def describe_receipt(receipt: Dict[str, float]) -> str:
            return (
                f"Subtotal={receipt['subtotal']:.2f}; "
                f"Discount={receipt['discount']:.2f}; "
                f"Total={receipt['total']:.2f}"
            )


        def sample_order() -> List[OrderItem]:
            return [
                OrderItem("Keyboard", 100.0, 1),
                OrderItem("Mouse", 50.0, 2),
                OrderItem("Cable", 10.0, 3),
            ]


        def main() -> str:
            items = sample_order()
            receipt = build_receipt(items, "gold", "SAVE10")
            return describe_receipt(receipt)


        if __name__ == "__main__":
            print(main())
        '''
    ).strip() + "\n"


def _make_project_tests() -> str:
    return textwrap.dedent(
        '''
        from order_project import (
            OrderItem,
            compute_subtotal,
            compute_discount_rate,
            compute_discounted_total,
            build_receipt,
            main,
        )


        def test_subtotal():
            items = [
                OrderItem("A", 10.0, 2),
                OrderItem("B", 5.0, 3),
            ]
            assert compute_subtotal(items) == 35.0


        def test_discount_rate_gold_save10():
            assert compute_discount_rate("gold", "SAVE10") == 0.25


        def test_discounted_total_gold_save10():
            items = [
                OrderItem("Keyboard", 100.0, 1),
                OrderItem("Mouse", 50.0, 2),
                OrderItem("Cable", 10.0, 3),
            ]
            # subtotal = 230.0, capped discount = 25%, total = 172.5
            assert compute_discounted_total(items, "gold", "SAVE10") == 172.5


        def test_discounted_total_silver_save5():
            items = [
                OrderItem("Desk", 200.0, 1),
            ]
            # subtotal = 200, discount = 8% + 5% = 13%, total = 174
            assert compute_discounted_total(items, "silver", "SAVE5") == 174.0


        def test_build_receipt():
            items = [
                OrderItem("Desk", 200.0, 1),
            ]
            receipt = build_receipt(items, "silver", "SAVE5")
            assert receipt["subtotal"] == 200.0
            assert receipt["discount"] == 26.0
            assert receipt["total"] == 174.0


        def test_main_runs():
            text = main()
            assert "Subtotal=" in text
            assert "Discount=" in text
            assert "Total=" in text
        '''
    ).strip() + "\n"


def _build_task_from_gap(source_with_gap: str) -> Dict[str, Any]:
    return {
        "task_id": "codereval_like_full_001",
        "lang": "python",
        "entry_function": "compute_discounted_total",
        "signature": (
            "def compute_discounted_total("
            "items: List[OrderItem], customer_tier: str, coupon_code: str | None = None"
            ") -> float:"
        ),
        "docstring": (
            "Restore the missing function compute_discounted_total in a project-sized Python file. "
            "The function must use existing helpers compute_subtotal(items) and "
            "compute_discount_rate(customer_tier, coupon_code), then apply the discount "
            "and return a rounded float with 2 decimals."
        ),
        "context_blocks": [
            "You are patching a missing function inside a larger existing codebase.",
            "Do not rewrite the whole file. Return only the target function implementation.",
            "The function must integrate with existing helpers and existing data model.",
            "Use compute_subtotal(items).",
            "Use compute_discount_rate(customer_tier, coupon_code).",
            "Return round(subtotal * (1 - rate), 2).",
            "Project source with the gap is below:",
            source_with_gap,
        ],
        "runnable_level": "project_runnable",
        "project_context": {
            "target_file": "order_project.py",
            "constraints": [
                "must_define:compute_discounted_total",
                "must_be_python",
                "deterministic_only",
                "must_call:compute_subtotal",
                "must_call:compute_discount_rate",
                "no_network",
                "no_file_io",
                "return_rounded_float_2dp",
            ],
            "tests": [
                {
                    "name": "discounted_total_gold_save10",
                    "call": "compute_discounted_total(sample_items(), 'gold', 'SAVE10')",
                    "expected": 172.5,
                }
            ],
        },
    }


def _make_inline_exec_harness_for_agent_validation() -> str:
    return textwrap.dedent(
        '''
        from order_project import OrderItem

        def sample_items():
            return [
                OrderItem("Keyboard", 100.0, 1),
                OrderItem("Mouse", 50.0, 2),
                OrderItem("Cable", 10.0, 3),
            ]
        '''
    ).strip()


def _patch_missing_function(source_with_gap: str, generated_function_code: str) -> str:
    pattern = re.compile(
        r"def compute_discounted_total\([\s\S]*?raise NotImplementedError\(\"MISSING_TARGET_FUNCTION\"\)\n",
        flags=re.MULTILINE,
    )
    m = pattern.search(source_with_gap)
    if not m:
        raise RuntimeError("Cannot locate target gap in source file.")
    replacement = generated_function_code.strip() + "\n"
    return source_with_gap[:m.start()] + replacement + source_with_gap[m.end():]


def _run_pytest(project_dir: Path) -> Dict[str, Any]:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    proc = subprocess.run(
        cmd,
        cwd=str(project_dir),
        capture_output=True,
        text=True,
    )
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "status": "pass" if proc.returncode == 0 else "fail",
    }


def _import_and_probe(project_file: Path) -> Dict[str, Any]:
    import sys
    import importlib.util

    module_name = "order_project"

    spec = importlib.util.spec_from_file_location(module_name, str(project_file))
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to build module spec for patched project file.")

    mod = importlib.util.module_from_spec(spec)

    # Critical: register module before exec_module(),
    # otherwise dataclass + postponed annotations may fail.
    sys.modules[module_name] = mod

    try:
        spec.loader.exec_module(mod)
    except Exception:
        # cleanup broken partially-loaded module
        sys.modules.pop(module_name, None)
        raise

    items = [
        mod.OrderItem("Keyboard", 100.0, 1),
        mod.OrderItem("Mouse", 50.0, 2),
        mod.OrderItem("Cable", 10.0, 3),
    ]
    value = mod.compute_discounted_total(items, "gold", "SAVE10")
    return {
        "probe_call": "compute_discounted_total(items, 'gold', 'SAVE10')",
        "probe_result": value,
    }


def main() -> int:
    try:
        p("[Step 0] Repo root", str(REPO_ROOT))

        config_path = REPO_ROOT / "configs" / "default.yaml"
        config = _load_yaml(config_path)
        p("[Step 1] Loaded config path", str(config_path))
        p("[Step 1.1] Config content", config)

        run_pipeline = _resolve_pipeline()
        p("[Step 2] Resolved pipeline entry", "beacon_system.pipeline.run_pipeline")

        with tempfile.TemporaryDirectory(prefix="beacon_codereval_like_") as tmp:
            tmp_dir = Path(tmp)
            project_dir = tmp_dir / "project"
            project_dir.mkdir(parents=True, exist_ok=True)

            source_with_gap = _make_large_project_source()
            tests_code = _make_project_tests()

            project_file = project_dir / "order_project.py"
            test_file = project_dir / "test_order_project.py"

            _write_text(project_file, source_with_gap)
            _write_text(test_file, tests_code)

            p("[Step 3] Temporary project dir", str(project_dir))
            p("[Step 3.1] Original large source with missing function", source_with_gap)
            p("[Step 3.2] Project tests", tests_code)

            # Show baseline failure before patching
            baseline = _run_pytest(project_dir)
            p("[Step 4] Baseline pytest result before agent patch", baseline)

            task = _build_task_from_gap(source_with_gap)
            p("[Step 5] Built agent task", task)

            # Small helper harness for the smoke exec path inside run_pipeline
            # This is only to make the inline smoke exec test richer if needed.
            inline_exec_harness = _make_inline_exec_harness_for_agent_validation()
            p("[Step 5.1] Inline exec harness (for understanding only)", inline_exec_harness)

            result = run_pipeline(task=task, config=config)
            p("[Step 6] Pipeline returned", result)

            run_dir = Path(result["run_dir"])
            p("[Step 6.1] Run dir", str(run_dir))

            task_json = _read_text(run_dir / "task.json")
            ir_json = _read_text(run_dir / "ir.json")
            constraints_json = _read_text(run_dir / "constraints.json")
            code_round1 = _read_text(run_dir / "code_round1.py")
            verifier_json = _read_text(run_dir / "verifier_round1.json")
            exec_json = _read_text(run_dir / "exec_round1.json")
            adapter_snapshot_json = _read_text(run_dir / "adapter_snapshot.json")

            p("[Step 7] task.json", task_json)
            p("[Step 7.1] ir.json", ir_json)
            p("[Step 7.2] constraints.json", constraints_json)
            p("[Step 7.3] code_round1.py (raw generated content)", code_round1)
            p("[Step 7.4] verifier_round1.json", verifier_json)
            p("[Step 7.5] exec_round1.json", exec_json)
            p("[Step 7.6] adapter_snapshot.json", adapter_snapshot_json)

            extracted_fn = _extract_function_code(code_round1, "compute_discounted_total")
            p("[Step 8] Extracted target function code", extracted_fn)

            # Validate generated function syntax before patching
            ast.parse(extracted_fn)
            p("[Step 8.1] AST parse of extracted function", "OK")

            patched_source = _patch_missing_function(source_with_gap, extracted_fn)
            p("[Step 9] Patched full source", patched_source)

            _write_text(project_file, patched_source)
            p("[Step 9.1] Wrote patched source file", str(project_file))

            probe = _import_and_probe(project_file)
            p("[Step 10] Direct import/probe result", probe)

            after_patch = _run_pytest(project_dir)
            p("[Step 11] Pytest result after agent patch", after_patch)

            final_summary = {
                "baseline_status": baseline["status"],
                "pipeline_status": result.get("status"),
                "pipeline_mode": result.get("mode"),
                "verifier_ok": result.get("verifier_ok"),
                "exec_status": result.get("exec_status"),
                "patched_probe_result": probe["probe_result"],
                "pytest_after_patch_status": after_patch["status"],
                "pytest_returncode": after_patch["returncode"],
                "run_dir": str(run_dir),
                "project_dir": str(project_dir),
            }
            p("[Step 12] Final summary", final_summary)

            if baseline["status"] != "fail":
                raise RuntimeError("Baseline project was expected to fail before patching, but it did not.")

            if result.get("status") != "ok":
                raise RuntimeError(f"Pipeline status is not ok: {result.get('status')}")

            if after_patch["status"] != "pass":
                raise RuntimeError("Patched project tests did not pass.")

            print("\n" + "=" * 100)
            print("CODEREVAL_LIKE_FULL_TEST_OK")
            print("=" * 100)
            return 0

    except Exception as e:
        print("\n" + "=" * 100)
        print("[FATAL] Full codereval-like test failed")
        print(repr(e))
        traceback.print_exc()
        print("=" * 100)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())