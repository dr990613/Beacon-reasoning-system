# src/beacon_system/llm/config.py
# -*- coding: utf-8 -*-

"""
LLM Config

- The ONLY module allowed to read environment variables for LLM connectivity.
- Merge priority:
  1) defaults
  2) YAML config dict (prefer `model`, compatible with `llm`)
  3) environment variables

Supported env keys:
- LITELLM_BASE_URL
- LITELLM_API_KEY
- MODEL_NAME

Also supports yaml placeholder style:
- ${env:VAR_NAME}
- ${env:VAR_NAME,default_value}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional
import os
import re


_ENV_PATTERN = re.compile(r"^\$\{env:([^,}]+)(?:,([^}]*))?\}$")


def _resolve_env_placeholder(value: Any) -> Any:
    """
    Resolve strings like:
    - ${env:MODEL_BASE_URL}
    - ${env:MODEL_BASE_URL,https://openrouter.ai/api/v1}
    """
    if not isinstance(value, str):
        return value

    text = value.strip()
    m = _ENV_PATTERN.match(text)
    if not m:
        return value

    env_name = (m.group(1) or "").strip()
    default_value = m.group(2)
    env_value = os.getenv(env_name)

    if env_value is not None and str(env_value).strip():
        return env_value.strip()

    if default_value is not None:
        return default_value.strip()

    return ""


@dataclass(frozen=True)
class ModelConfig:
    """
    Unified model connectivity/config contract.
    """
    base_url: str = "https://openrouter.ai/api/v1"
    api_key: str = "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"
    model_name: str = "qwen/qwen3-32b"
    timeout_s: int = 120
    params: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 2048,
    })

    @staticmethod
    def from_sources(model_dict: Optional[Mapping[str, Any]] = None) -> "ModelConfig":
        model_dict = dict(model_dict or {})

        base_url = _resolve_env_placeholder(model_dict.get("base_url") or ModelConfig.base_url)
        api_key = _resolve_env_placeholder(model_dict.get("api_key") or ModelConfig.api_key)
        model_name = _resolve_env_placeholder(model_dict.get("model_name") or ModelConfig.model_name)
        timeout_s = int(_resolve_env_placeholder(model_dict.get("timeout_s") or ModelConfig.timeout_s))

        params = dict(ModelConfig().params)
        params.update(dict(model_dict.get("params") or {}))

        # env overrides (ONLY here)
        base_url = os.getenv("LITELLM_BASE_URL", str(base_url))
        api_key = os.getenv("LITELLM_API_KEY", str(api_key))
        model_name = os.getenv("MODEL_NAME", str(model_name))

        if api_key is None:
            api_key = ""

        return ModelConfig(
            base_url=str(base_url),
            api_key=str(api_key),
            model_name=str(model_name),
            timeout_s=int(timeout_s),
            params=params,
        )

    @staticmethod
    def from_config_dict(config: Optional[Mapping[str, Any]] = None) -> "ModelConfig":
        config = dict(config or {})
        model_dict = config.get("model")
        if model_dict is None:
            model_dict = config.get("llm")
        return ModelConfig.from_sources(model_dict=model_dict)

    def validate(self) -> None:
        if not self.base_url.strip():
            raise ValueError("Missing model base_url in config['model'] or environment.")
        if not self.model_name.strip():
            raise ValueError("Missing model model_name in config['model'] or environment.")
        if not self.api_key.strip():
            raise ValueError("Missing model api_key in config['llm'] or config['model'].")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")

    def to_request_kwargs(self) -> Dict[str, Any]:
        return {
            "base_url": self.base_url,
            "api_key": self.api_key,
            "model": self.model_name,
            "timeout": self.timeout_s,
            "params": dict(self.params),
        }

    def masked(self) -> Dict[str, Any]:
        masked_key = ""
        if self.api_key:
            if len(self.api_key) <= 8:
                masked_key = "***"
            else:
                masked_key = f"{self.api_key[:6]}***{self.api_key[-4:]}"
        return {
            "base_url": self.base_url,
            "api_key": masked_key,
            "model_name": self.model_name,
            "timeout_s": self.timeout_s,
            "params": dict(self.params),
        }