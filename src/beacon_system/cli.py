# src/beacon_system/cli.py
# -*- coding: utf-8 -*-

"""
CLI entrypoint

Responsibilities (MUST stay minimal):
- Parse args
- Load YAML config
- Build RunConfig / ModelConfig
- Create adapter + runtime + llm_client
- Call pipeline.run()

Hard rules:
- No business logic (no reasoning, no verification, no runtime details)
- No env access here (env only in llm/config.py)
- No adapter registry; only explicit if/elif selection
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, Optional

import yaml

from .llm.config import ModelConfig
from .llm.client import LLMClient
from .pipeline import run as pipeline_run


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Invalid yaml root type: {type(data)}")
    return data


def _merge_dict(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Shallow merge only (keep minimal). Nested merges are handled explicitly per-section.
    """
    out = dict(base)
    if override:
        out.update(dict(override))
    return out


def _build_run_config_dict(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """
    Minimal assembly of run config dict; types.RunConfig construction can happen in pipeline
    (or you can add a RunConfig.from_sources later).
    """
    run_cfg = dict(cfg.get("run") or {})
    reader_cfg = dict(cfg.get("reader") or {})
    model_cfg = dict(cfg.get("model") or {})
    adapter_cfg = dict(cfg.get("adapter") or {})

    # CLI overrides (ONLY run.* by rule)
    if args.seed is not None:
        run_cfg["seed"] = int(args.seed)
    if args.max_rounds is not None:
        run_cfg["max_rounds"] = int(args.max_rounds)
    if args.use_verifier is not None:
        run_cfg["use_verifier"] = bool(args.use_verifier)
    if args.outputs_dir is not None:
        run_cfg["outputs_dir"] = str(args.outputs_dir)

    # Adapter selection via CLI (optional)
    if args.adapter is not None:
        adapter_cfg["name"] = str(args.adapter)
    if args.repo_root is not None:
        adapter_cfg.setdefault("params", {})
        adapter_cfg["params"]["repo_root"] = str(args.repo_root)
    if args.target_file is not None:
        adapter_cfg.setdefault("params", {})
        adapter_cfg["params"]["target_file"] = str(args.target_file)
    if args.target_qualname is not None:
        adapter_cfg.setdefault("params", {})
        adapter_cfg["params"]["target_qualname"] = str(args.target_qualname)
    if args.spec is not None:
        adapter_cfg.setdefault("params", {})
        adapter_cfg["params"]["spec"] = str(args.spec)

    return {
        "run": run_cfg,
        "reader": reader_cfg,
        "model": model_cfg,
        "adapter": adapter_cfg,
    }


def _create_adapter_and_runtime(adapter_cfg: Dict[str, Any]):
    """
    No registry: explicit selection only.
    """
    name = (adapter_cfg.get("name") or "localrepo").strip()
    params = dict(adapter_cfg.get("params") or {})

    if name == "localrepo":
        from .adapters.localrepo.task_adapter import LocalRepoTaskAdapter
        from .adapters.localrepo.runtime import LocalRepoRuntimeAdapter

        task_adapter = LocalRepoTaskAdapter(
            repo_root=params.get("repo_root", "."),
            target_file=params.get("target_file"),
            target_qualname=params.get("target_qualname"),
            spec=params.get("spec", ""),
            context=params.get("context") or {},
            meta=params.get("meta") or {},
        )
        runtime = LocalRepoRuntimeAdapter(
            repo_root=params.get("repo_root", "."),
            run_cmd=params.get("run_cmd", "pytest -q"),
            work_dir=params.get("work_dir", None),
        )
        return task_adapter, runtime

    raise ValueError(f"Unknown adapter name: {name}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="beacon-system", add_help=True)

    p.add_argument("--config", type=str, default="configs/default.yaml", help="Path to yaml config")

    # Run overrides (ONLY run.* allowed)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--use-verifier", action="store_true", default=None)
    p.add_argument("--no-verifier", dest="use_verifier", action="store_false")
    p.add_argument("--outputs-dir", type=str, default=None)

    # Adapter selection (optional)
    p.add_argument("--adapter", type=str, default=None, help="Adapter name (default: localrepo)")
    p.add_argument("--repo-root", type=str, default=None)
    p.add_argument("--target-file", type=str, default=None)
    p.add_argument("--target-qualname", type=str, default=None)
    p.add_argument("--spec", type=str, default=None)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = _load_yaml(args.config)
    cfg2 = _build_run_config_dict(cfg, args)

    # Build ModelConfig (env override happens ONLY inside ModelConfig.from_sources)
    model_cfg = ModelConfig.from_sources(cfg2.get("model") or {})

    # Create injected client
    llm_client = LLMClient(model_cfg)

    # Adapter + runtime
    task_adapter, runtime = _create_adapter_and_runtime(cfg2.get("adapter") or {})

    # Run pipeline (pipeline owns orchestration)
    pipeline_run(
        run_cfg_dict=cfg2,
        task_adapter=task_adapter,
        runtime=runtime,
        llm=llm_client,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())