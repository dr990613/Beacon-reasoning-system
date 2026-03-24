# -*- coding: utf-8 -*-

"""
LLM config loader.

Design goals:
- Extremely simple and explicit
- Read directly from yaml
- No env indirection
- Easy to switch model/provider by editing config file only
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict
import yaml


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("YAML config root must be a dict.")

    return data


def _to_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = ""
    model_name: str = "qwen/qwen3-coder-next"
    timeout_sec: int = 120
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 4096

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_llm_config(config_path: str | Path) -> LLMConfig:
    """
    Load llm config from yaml only.
    """
    raw = _read_yaml(config_path)
    llm = raw.get("llm", {})

    if not isinstance(llm, dict):
        raise ValueError("Config field 'llm' must be a dict.")

    return LLMConfig(
        base_url=str(llm.get("base_url", "https://openrouter.ai/api/v1")).strip(),
        api_key=str(llm.get("api_key", "")).strip(),
        model_name=str(llm.get("model_name", "qwen/qwen3-coder-next")).strip(),
        timeout_sec=_to_int(llm.get("timeout_sec", 120), 120),
        max_retries=_to_int(llm.get("max_retries", 2), 2),
        temperature=_to_float(llm.get("temperature", 0.0), 0.0),
        max_tokens=_to_int(llm.get("max_tokens", 4096), 2048),
    )


def load_runtime_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load full yaml config for other modules.
    """
    return _read_yaml(config_path)


def resolve_output_dir(config_path: str | Path, default: str = "outputs") -> str:
    raw = _read_yaml(config_path)
    artifacts = raw.get("artifacts", {})
    if not isinstance(artifacts, dict):
        return default
    return str(artifacts.get("output_dir", default))