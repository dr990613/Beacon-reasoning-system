# baseline_codegen/config_baseline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _default_project_root() -> Path:
    """
    Infer repository root from this file location.

    baseline_codegen/config_baseline.py
    -> repo root = parent of baseline_codegen
    """
    return Path(__file__).resolve().parent.parent


@dataclass
class ModelConfig:
    """
    Baseline LLM model configuration.
    """
    model_name: str = "qwen/qwen3-coder-next"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2048
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
    Runtime configuration for baseline batch execution.

    Backward compatibility:
    - input_json_path and output_json_path are preserved for old scripts.
    - New fields are added for Python/Java split input and pass@k style generation.
    """
    project_root: str = ""
    language: str = "java"   # python | java

    # New: explicit per-language input paths
    python_input_json_path: str = ""
    java_input_json_path: str = ""

    # Backward-compatible legacy single-input path
    input_json_path: str = ""

    # Output organization
    output_dir: str = ""
    output_json_path: str = ""
    merged_output_json_path: str = ""

    # Batch control
    task_limit: Optional[int] = None
    start_index: int = 0
    num_passes: int = 1

    def resolved_project_root(self) -> Path:
        return Path(self.project_root).resolve()

    def resolved_output_dir(self) -> Path:
        return _resolve_path(self.output_dir, self.resolved_project_root())

    def resolved_python_input_json_path(self) -> Path:
        return _resolve_path(self.python_input_json_path, self.resolved_project_root())

    def resolved_java_input_json_path(self) -> Path:
        return _resolve_path(self.java_input_json_path, self.resolved_project_root())

    def resolved_input_json_path(self) -> Path:
        """
        Compatible resolver:
        1. If legacy input_json_path is set, use it.
        2. Otherwise choose by language.
        """
        if self.input_json_path:
            return _resolve_path(self.input_json_path, self.resolved_project_root())

        lang = (self.language or "python").strip().lower()
        if lang == "java":
            return self.resolved_java_input_json_path()
        return self.resolved_python_input_json_path()

    def resolved_output_json_path(self) -> Path:
        return _resolve_path(self.output_json_path, self.resolved_project_root())

    def resolved_merged_output_json_path(self) -> Path:
        return _resolve_path(self.merged_output_json_path, self.resolved_project_root())


@dataclass
class BaselineConfig:
    """
    Root config object for the baseline pipeline.
    """
    model: ModelConfig = field(default_factory=ModelConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    run: RunConfig = field(default_factory=RunConfig)


def _read_env_str(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value if value else default


def _read_env_str_any(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip():
            return value.strip()
    return default


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


def _read_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _resolve_path(path_str: str, project_root: Path) -> Path:
    """
    Resolve relative paths against project_root.
    """
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _default_run_paths(project_root: Path) -> dict:
    output_dir = project_root / "outputs" / "baseline"

    return {
        "python_input_json_path": str(project_root / "benchmarks" / "CoderEval" / "CoderEval4Python.json"),
        "java_input_json_path": str(project_root / "benchmarks" / "CoderEval" / "CoderEval4Java.json"),
        "input_json_path": str(project_root / "benchmarks" / "CoderEval" / "CoderEval4Python.json"),  # legacy default
        "output_dir": str(output_dir),
        "output_json_path": str(output_dir / "baseline_results.json"),  # legacy default
        "merged_output_json_path": str(output_dir / "baseline_results_merged.jsonl"),
    }


def load_config() -> BaselineConfig:
    """
    Load configuration from environment variables.

    Model envs:
    - BASELINE_MODEL_NAME
    - MODEL_NAME
    - BASELINE_API_KEY
    - OPENROUTER_API_KEY
    - OPENAI_API_KEY
    - BASELINE_BASE_URL
    - MODEL_BASE_URL
    - OPENAI_BASE_URL
    - BASELINE_TEMPERATURE
    - BASELINE_MAX_TOKENS
    - BASELINE_TIMEOUT

    Generation envs:
    - BASELINE_SYSTEM_PROMPT
    - BASELINE_REQUIRE_CODE_ONLY
    - BASELINE_MIN_CODE_LENGTH
    - BASELINE_MAX_RETRIES

    Run envs:
    - BASELINE_PROJECT_ROOT
    - BASELINE_LANGUAGE
    - BASELINE_PYTHON_INPUT_JSON
    - BASELINE_JAVA_INPUT_JSON
    - BASELINE_INPUT_JSON              # legacy
    - BASELINE_OUTPUT_DIR
    - BASELINE_OUTPUT_JSON             # legacy
    - BASELINE_MERGED_OUTPUT_JSON
    - BASELINE_TASK_LIMIT
    - BASELINE_START_INDEX
    - BASELINE_NUM_PASSES
    """
    project_root = Path(
        _read_env_str("BASELINE_PROJECT_ROOT", str(_default_project_root()))
    ).resolve()

    defaults = _default_run_paths(project_root)

    task_limit_raw = _read_env_str("BASELINE_TASK_LIMIT")
    task_limit = int(task_limit_raw) if task_limit_raw is not None else None

    cfg = BaselineConfig(
        model=ModelConfig(
            model_name=_read_env_str_any(
                "BASELINE_MODEL_NAME",
                "MODEL_NAME",
                default="qwen/qwen3-coder-next",
            ),
            api_key=_read_env_str_any(
                "BASELINE_API_KEY",
                "OPENROUTER_API_KEY",
                "OPENAI_API_KEY",
                default=None,
            ),
            base_url=_read_env_str_any(
                "BASELINE_BASE_URL",
                "MODEL_BASE_URL",
                "OPENAI_BASE_URL",
                default=None,
            ),
            temperature=_read_env_float("BASELINE_TEMPERATURE", 0.0),
            max_tokens=_read_env_int("BASELINE_MAX_TOKENS", 2048),
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
            require_code_only=_read_env_bool("BASELINE_REQUIRE_CODE_ONLY", True),
            min_code_length=_read_env_int("BASELINE_MIN_CODE_LENGTH", 10),
            max_retries=_read_env_int("BASELINE_MAX_RETRIES", 2),
        ),
        run=RunConfig(
            project_root=str(project_root),
            language=_read_env_str("BASELINE_LANGUAGE", "python").lower(),
            python_input_json_path=_read_env_str(
                "BASELINE_PYTHON_INPUT_JSON",
                defaults["python_input_json_path"],
            ),
            java_input_json_path=_read_env_str(
                "BASELINE_JAVA_INPUT_JSON",
                defaults["java_input_json_path"],
            ),
            input_json_path=_read_env_str(
                "BASELINE_INPUT_JSON",
                defaults["input_json_path"],
            ),
            output_dir=_read_env_str(
                "BASELINE_OUTPUT_DIR",
                defaults["output_dir"],
            ),
            output_json_path=_read_env_str(
                "BASELINE_OUTPUT_JSON",
                defaults["output_json_path"],
            ),
            merged_output_json_path=_read_env_str(
                "BASELINE_MERGED_OUTPUT_JSON",
                defaults["merged_output_json_path"],
            ),
            task_limit=task_limit,
            start_index=_read_env_int("BASELINE_START_INDEX", 0),
            num_passes=_read_env_int("BASELINE_NUM_PASSES", 1),
        ),
    )
    return cfg