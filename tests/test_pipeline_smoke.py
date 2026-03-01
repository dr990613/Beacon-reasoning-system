# tests/test_pipeline_smoke.py
# -*- coding: utf-8 -*-

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


from beacon_system.pipeline import run as pipeline_run
from beacon_system.llm.config import ModelConfig
from beacon_system.adapters.localrepo.task_adapter import LocalRepoTaskAdapter
from beacon_system.adapters.localrepo.runtime import LocalRepoRuntimeAdapter


@dataclass
class DummyLLM:
    """
    Local silent LLM stub (no network):
    - Round1: return wrong code (to trigger verifier -> revise)
    - Round2+: return correct code
    - Also prints which round it is (observable).
    """
    cfg: ModelConfig
    calls: int = 0

    def chat(self, messages: List[Dict[str, str]], *, model: Optional[str] = None, **kwargs: Any) -> str:
        self.calls += 1
        round_id = self.calls
        print(f"[DummyLLM] chat called, round={round_id}, model={model or self.cfg.model_name}")

        # Round 1: wrong code -> should fail tests (and ideally verifier may still pass structurally)
        if round_id == 1:
            return (
                "def foo(x):\n"
                "    return x\n"
            )

        # Round 2+: correct
        return (
            "def foo(x):\n"
            "    return x + 1\n"
        )

    def complete(self, prompt: str, *, model: Optional[str] = None, **kwargs: Any) -> str:
        return self.chat([{"role": "user", "content": prompt}], model=model, **kwargs)


def _write(path: str, s: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(s)


def _list_run_dirs(outputs_dir: str) -> List[str]:
    ds = [os.path.join(outputs_dir, d) for d in os.listdir(outputs_dir)]
    ds = [d for d in ds if os.path.isdir(d)]
    return sorted(ds)


def _exists(run_dir: str, name: str) -> bool:
    return os.path.exists(os.path.join(run_dir, name))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def test_pipeline_smoke_localrepo_dummy_llm_with_revision_loop_and_observability():
    """
    End-to-end smoke test (no real LLM calls) with 2 rounds:
      - Round1: DummyLLM returns wrong code
      - Round2: DummyLLM returns corrected code
    We print key checkpoints and assert:
      - run dir exists
      - required artifacts exist
      - exec_round1 exists (should fail)
      - exec_round2 exists (should pass)
    """
    with tempfile.TemporaryDirectory() as repo_root, tempfile.TemporaryDirectory() as outputs_dir:
        # 1) Create minimal repo
        _write(
            os.path.join(repo_root, "my_module.py"),
            "def foo(x):\n"
            "    # placeholder implementation\n"
            "    raise NotImplementedError\n",
        )
        _write(
            os.path.join(repo_root, "test_my_module.py"),
            "from my_module import foo\n\n"
            "def test_foo():\n"
            "    assert foo(1) == 2\n",
        )

        print(f"\n[Smoke] repo_root   = {repo_root}")
        print(f"[Smoke] outputs_dir = {outputs_dir}")

        # 2) Build adapter/runtime
        task_adapter = LocalRepoTaskAdapter(
            repo_root=repo_root,
            target_file="my_module.py",
            target_qualname="foo",
            spec="Implement foo(x) that returns x + 1",
            context={},
            meta={"id": "smoke-localrepo", "lang": "python", "level": "function"},
        )
        runtime = LocalRepoRuntimeAdapter(
            repo_root=repo_root,
            run_cmd="pytest -q",
            work_dir=None,
        )

        # 3) Dummy LLM (no network)
        llm = DummyLLM(ModelConfig())

        # 4) Minimal config dict (2 rounds to test revise loop)
        run_cfg_dict = {
            "run": {
                "seed": 0,
                "max_rounds": 2,          # ✅ revision loop
                "use_verifier": True,
                "outputs_dir": outputs_dir,
            },
            "reader": {
                "enable_global": True,
                "validation_filter": True,
                "max_local_nodes": 400,
                "max_global_inline": None,
            },
            "model": {},
            "adapter": {
                "name": "localrepo",
                "params": {
                    "repo_root": repo_root,
                    "target_file": "my_module.py",
                    "target_qualname": "foo",
                    "spec": "Implement foo(x) that returns x + 1",
                    "run_cmd": "pytest -q",
                },
            },
        }

        # 5) Run pipeline
        pipeline_run(
            run_cfg_dict=run_cfg_dict,
            task_adapter=task_adapter,
            runtime=runtime,
            llm=llm,
            memory=None,
        )

        # 6) Find run_dir
        run_dirs = _list_run_dirs(outputs_dir)
        assert run_dirs, "No run directory created under outputs_dir"
        run_dir = run_dirs[-1]
        print(f"[Smoke] run_dir = {run_dir}")

        # 7) Check base artifacts
        base_artifacts = [
            "config.yaml",
            "adapter_snapshot.json",
            "task.json",
            "ir.json",
            "constraints.json",
        ]
        for name in base_artifacts:
            ok = _exists(run_dir, name)
            print(f"[Check] exists {name}: {ok}")
            assert ok, f"Missing artifact: {name}"

        # 8) Round artifacts existence
        round_artifacts = [
            "code_round1.py",
            "verifier_round1.json",
            "exec_round1.json",
            "code_round2.py",
            "verifier_round2.json",
            "exec_round2.json",
        ]
        for name in round_artifacts:
            ok = _exists(run_dir, name)
            print(f"[Check] exists {name}: {ok}")
            assert ok, f"Missing artifact: {name}"

        # 9) Inspect round results
        exec1 = _load_json(os.path.join(run_dir, "exec_round1.json"))
        exec2 = _load_json(os.path.join(run_dir, "exec_round2.json"))
        rep1 = _load_json(os.path.join(run_dir, "verifier_round1.json"))
        rep2 = _load_json(os.path.join(run_dir, "verifier_round2.json"))

        print("\n[Round1] verifier.ok =", rep1.get("ok"), "exec.status =", exec1.get("status"), "rc =", exec1.get("return_code"))
        print("[Round2] verifier.ok =", rep2.get("ok"), "exec.status =", exec2.get("status"), "rc =", exec2.get("return_code"))

        # Round1 should fail runtime (wrong code)
        assert exec1.get("status") in ("fail", "error"), f"Expected round1 fail/error, got {exec1.get('status')}"

        # Round2 should pass runtime (fixed code)
        assert exec2.get("status") == "pass", f"Expected round2 pass, got {exec2.get('status')}"
        assert int(exec2.get("return_code", 1)) == 0, f"Non-zero round2 return_code: {exec2.get('return_code')}"

        # 10) trace file (optional) - print whether present
        trace1 = os.path.join(run_dir, "exec_round1.trace.txt")
        trace2 = os.path.join(run_dir, "exec_round2.trace.txt")
        print(f"\n[Trace] round1 trace exists: {os.path.exists(trace1)}")
        print(f"[Trace] round2 trace exists: {os.path.exists(trace2)}")
        if os.path.exists(trace1):
            with open(trace1, "r", encoding="utf-8") as f:
                t = f.read()
            print("[Trace] round1 trace preview:\n", t[:500])