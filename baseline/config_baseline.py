# baseline_codegen/config_baseline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    """
    Baseline LLM model configuration.
    """
    model_name: str = "qwen/qwen3-32b"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 120


@dataclass
class GenerationConfig:
    """
    Generation behavior for baseline code generation.
    """
    system_prompt: str = (
        "You are a coding model. "
        "Given a programming task, output only the final code solution. "
        "Do not include explanations, markdown fences, or extra text."
    )
    require_code_only: bool = True
    min_code_length: int = 10
    max_retries: int = 2


@dataclass
class RunConfig:
    """
    Runtime configuration for batch execution.
    """
    input_json_path: str = "./benchmarks/CoderEval/CoderEval4Python.json"
    output_json_path: str = "./outputs/baseline_results.json"
    task_limit: Optional[int] = None
    start_index: int = 0


@dataclass
class BaselineConfig:
    """
    Root config object for the baseline pipeline.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _read_env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name, default)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _read_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _read_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return float(raw.strip())


def load_config() -> BaselineConfig:
    """
    Load configuration from environment variables.

    Supported environment variables:
    - BASELINE_MODEL_NAME
    - BASELINE_API_KEY
    - BASELINE_BASE_URL
    - BASELINE_TEMPERATURE
    - BASELINE_MAX_TOKENS
    - BASELINE_TIMEOUT
    - BASELINE_SYSTEM_PROMPT
    - BASELINE_REQUIRE_CODE_ONLY
    - BASELINE_MIN_CODE_LENGTH
    - BASELINE_MAX_RETRIES
    - BASELINE_INPUT_JSON
    - BASELINE_OUTPUT_JSON
    - BASELINE_TASK_LIMIT
    - BASELINE_START_INDEX
    """
    require_code_only_raw = _read_env_str("BASELINE_REQUIRE_CODE_ONLY", "true")
    require_code_only = require_code_only_raw.lower() in {"1", "true", "yes", "y"}

    task_limit_raw = _read_env_str("BASELINE_TASK_LIMIT")
    task_limit = int(task_limit_raw) if task_limit_raw is not None else None

    cfg = BaselineConfig(
        model=ModelConfig(
            model_name=_read_env_str(
                "BASELINE_MODEL_NAME",
                "qwen/qwen3-32b",
            ),
            api_key=_read_env_str("BASELINE_API_KEY"),
            base_url=_read_env_str("BASELINE_BASE_URL"),
            temperature=_read_env_float("BASELINE_TEMPERATURE", 0.0),
            max_tokens=_read_env_int("BASELINE_MAX_TOKENS", 1024),
            timeout=_read_env_int("BASELINE_TIMEOUT", 120),
        ),
        generation=GenerationConfig(
            system_prompt=_read_env_str(
                "BASELINE_SYSTEM_PROMPT",
                (
                    "You are a coding model. "
                    "Given a programming task, output only the final code solution. "
                    "Do not include explanations, markdown fences, or extra text."
                ),
            ),
            require_code_only=require_code_only,
            min_code_length=_read_env_int("BASELINE_MIN_CODE_LENGTH", 10),
            max_retries=_read_env_int("BASELINE_MAX_RETRIES", 2),
        ),
        run=RunConfig(
            input_json_path=_read_env_str(
                "BASELINE_INPUT_JSON",
                "./benchmarks/CoderEval/CoderEval4Python.json",
            ),
            output_json_path=_read_env_str(
                "BASELINE_OUTPUT_JSON",
                "./outputs/baseline_results.json",
            ),
            task_limit=task_limit,
            start_index=_read_env_int("BASELINE_START_INDEX", 0),
        ),
    )
    return cfg