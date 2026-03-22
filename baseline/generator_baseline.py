# baseline_codegen/generator_baseline.py
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import json

from openai import OpenAI

from config_baseline import BaselineConfig
from schema import (
    CodeEvalTask,
    CodeOnlyPolicy,
    GenerationResult,
    ModelRequest,
    ModelResponse,
)


@dataclass
class BaselineGenerator:
    """
    Minimal baseline code generator.

    Responsibilities:
    1. Build prompt from CodeEvalTask
    2. Call OpenAI-compatible chat endpoint
    3. Extract code-only output
    4. Return GenerationResult in benchmark format
    """
    config: BaselineConfig

    def __post_init__(self) -> None:
        self.policy = CodeOnlyPolicy(
            min_code_length=self.config.generation.min_code_length,
            allow_markdown_fence=True,
        )
        self.client = self._build_client()

    def _build_client(self) -> OpenAI:
        """
        Build an OpenAI-compatible client.
        Works with OpenAI API, vLLM, LiteLLM proxy, etc.
        """
        kwargs: Dict[str, Any] = {
            "api_key": self.config.model.api_key or "EMPTY",
        }
        if self.config.model.base_url:
            kwargs["base_url"] = self.config.model.base_url
        return OpenAI(**kwargs)

    def build_request(self, task: CodeEvalTask) -> ModelRequest:
        """
        Convert one task into a standardized model request.
        """
        user_prompt = self._build_user_prompt(task)
        return ModelRequest(
            system_prompt=self.config.generation.system_prompt,
            user_prompt=user_prompt,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
        )

    def _build_user_prompt(self, task: CodeEvalTask) -> str:
        """
        Assemble a stable user prompt.
        Keep it simple: task content first, output constraint second.
        """
        parts = [
            task.prompt.strip(),
            "",
            "Requirements:",
            "1. Output only Python code.",
            "2. Do not include markdown fences.",
            "3. Do not include explanations or comments outside the code.",
            "4. Keep the implementation complete and runnable when possible.",
        ]
        return "\n".join(parts).strip()

    def call_model(self, request: ModelRequest) -> ModelResponse:
        """
        Call the chat model and normalize the result.
        """
        response = self.client.chat.completions.create(
            model=self.config.model.model_name,
            messages=request.to_chat_messages(),
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            timeout=self.config.model.timeout,
        )

        choice = response.choices[0]
        content = choice.message.content or ""
        finish_reason = getattr(choice, "finish_reason", None)

        raw = self._safe_model_dump(response)

        return ModelResponse(
            content=content,
            finish_reason=finish_reason,
            raw=raw,
        )

    def generate_code(self, task: CodeEvalTask) -> str:
        """
        Main generation path with light retry.
        Retry only when output is not code-like.
        """
        request = self.build_request(task)
        last_error: Optional[Exception] = None

        for attempt in range(self.config.generation.max_retries + 1):
            response = self.call_model(request)
            try:
                code = self.policy.validate_or_raise(response.content)
                return code
            except Exception as exc:
                last_error = exc
                request = self._build_retry_request(task, previous_output=response.content)

        raise ValueError(f"Failed to generate valid code for task={task.task_id}: {last_error}")

    def _build_retry_request(self, task: CodeEvalTask, previous_output: str) -> ModelRequest:
        """
        Stronger retry prompt when first output is not code-only.
        """
        retry_user_prompt = "\n".join(
            [
                task.prompt.strip(),
                "",
                "Your previous answer was rejected because it was not pure code.",
                "Now return ONLY the final Python code.",
                "Do not include explanations.",
                "Do not include markdown fences.",
                "Do not include any text before or after the code.",
                "",
                "Rejected output:",
                previous_output.strip() or "<empty>",
            ]
        ).strip()

        return ModelRequest(
            system_prompt=self.config.generation.system_prompt,
            user_prompt=retry_user_prompt,
            temperature=self.config.model.temperature,
            max_tokens=self.config.model.max_tokens,
        )

    def generate_result(self, task: CodeEvalTask) -> GenerationResult:
        """
        Generate one benchmark-formatted result item.
        """
        code = self.generate_code(task)
        return GenerationResult(
            _id=task.task_id,
            generate_results=[code],
        )

    @staticmethod
    def _safe_model_dump(response: Any) -> Dict[str, Any]:
        """
        Best-effort serialization for debugging/logging.
        """
        if hasattr(response, "model_dump"):
            try:
                return response.model_dump()
            except Exception:
                pass
        if hasattr(response, "dict"):
            try:
                return response.dict()
            except Exception:
                pass
        try:
            return json.loads(json.dumps(response, default=str))
        except Exception:
            return {"raw": str(response)}