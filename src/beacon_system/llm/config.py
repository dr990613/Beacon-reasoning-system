# src/beacon_system/llm/config.py
# -*- coding: utf-8 -*-

"""
LLM Config

- The ONLY module allowed to read environment variables for LLM connectivity.
- Merge priority:
  1) defaults
  2) YAML config dict (model.*)
  3) environment variables (LITELLM_BASE_URL / LITELLM_API_KEY / MODEL_NAME)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional
import os


@dataclass(frozen=True)
class ModelConfig:
    base_url: str = "http://a6k2.dgx:34000/v1"
    api_key: str = "sk-3ytNnX-OUrY4WQmGwJBmQA"  # MUST be empty by default. Provide via env or local .env
    model_name: str = "qwen3-32b"
    timeout_s: int = 120
    params: Dict[str, Any] = field(default_factory=lambda: {
        "temperature": 0.2,
        "top_p": 0.95,
        "max_tokens": 2048,
    })

    @staticmethod
    def from_sources(model_dict: Optional[Mapping[str, Any]] = None) -> "ModelConfig":
        """
        Construct ModelConfig by merging defaults + yaml(model_dict) + env overrides.

        model_dict is expected to be configs/default.yaml's `model:` mapping, e.g.:
          model:
            base_url: ...
            api_key: ...
            model_name: ...
            timeout_s: 120
            params: {temperature: 0.2, ...}
        """
        model_dict = dict(model_dict or {})

        # 1) defaults
        base_url = str(model_dict.get("base_url") or ModelConfig.base_url)
        api_key = str(model_dict.get("api_key") or ModelConfig.api_key)
        model_name = str(model_dict.get("model_name") or ModelConfig.model_name)
        timeout_s = int(model_dict.get("timeout_s") or ModelConfig.timeout_s)

        params = dict(ModelConfig().params)
        params.update(dict(model_dict.get("params") or {}))

        # 2) env overrides (ONLY here)
        base_url = os.getenv("LITELLM_BASE_URL", base_url)
        api_key = os.getenv("LITELLM_API_KEY", api_key)
        model_name = os.getenv("MODEL_NAME", model_name)

        # Safety: never force a non-empty default key in code
        if api_key is None:
            api_key = ""

        return ModelConfig(
            base_url=base_url,
            api_key=api_key,
            model_name=model_name,
            timeout_s=timeout_s,
            params=params,
        )