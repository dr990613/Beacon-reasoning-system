# -*- coding: utf-8 -*-

"""
Unified LLM client for Beacon system.

Design goals:
- Single chat entry
- Simple OpenAI-compatible client
- Explicit config contract: base_url + api_key + model_name
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import time

from openai import OpenAI

from .config import LLMConfig


@dataclass
class LLMResponse:
    text: str
    raw_response: Dict[str, Any]
    prompt_snapshot: str
    model_name: str
    usage: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LLMClient:
    """
    Unified chat client for OpenAI-compatible APIs.
    """

    def __init__(
        self,
        config: LLMConfig,
        *,
        debug_dir: Optional[str | Path] = None,
    ) -> None:
        self.config = config
        self.debug_dir = Path(debug_dir) if debug_dir else None

        self._client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )

    def chat(
        self,
        prompt: str,
        *,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> LLMResponse:
        messages = self._build_messages(prompt=prompt, system_prompt=system_prompt)
        prompt_snapshot = self._build_prompt_snapshot(messages)

        last_error: Optional[Exception] = None
        attempts = max(1, self.config.max_retries + 1)

        for attempt in range(1, attempts + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=self.config.temperature if temperature is None else temperature,
                    max_tokens=self.config.max_tokens if max_tokens is None else max_tokens,
                    timeout=self.config.timeout_sec,
                )

                text = self._extract_text(response)
                raw = self._safe_dump_response(response)
                usage = raw.get("usage") if isinstance(raw, dict) else None

                result = LLMResponse(
                    text=text,
                    raw_response=raw,
                    prompt_snapshot=prompt_snapshot,
                    model_name=self.config.model_name,
                    usage=usage if isinstance(usage, dict) else None,
                )

                self._write_debug_files(prompt_snapshot=prompt_snapshot, raw_response=raw)
                return result

            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break
                time.sleep(min(2 ** (attempt - 1), 4))

        raise RuntimeError(f"LLM chat failed after retries: {last_error}") from last_error

    @staticmethod
    def _build_messages(prompt: str, system_prompt: Optional[str]) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    @staticmethod
    def _build_prompt_snapshot(messages: List[Dict[str, str]]) -> str:
        parts: List[str] = []
        for msg in messages:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts).strip()

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            choice = response.choices[0]
            message = choice.message
            content = getattr(message, "content", "")

            if isinstance(content, str):
                return content.strip()

            if isinstance(content, list):
                chunks: List[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        chunks.append(str(item.get("text", "")))
                    elif hasattr(item, "text"):
                        chunks.append(str(item.text))
                return "".join(chunks).strip()

            return str(content).strip()

        except Exception as exc:
            raise RuntimeError(f"Failed to parse LLM response text: {exc}") from exc

    @staticmethod
    def _safe_dump_response(response: Any) -> Dict[str, Any]:
        if hasattr(response, "model_dump"):
            return response.model_dump()
        if hasattr(response, "to_dict"):
            return response.to_dict()
        try:
            return json.loads(json.dumps(response, default=str))
        except Exception:
            return {"raw": str(response)}

    def _write_debug_files(self, *, prompt_snapshot: str, raw_response: Dict[str, Any]) -> None:
        if not self.debug_dir:
            return

        self.debug_dir.mkdir(parents=True, exist_ok=True)
        ts = str(int(time.time() * 1000))

        prompt_path = self.debug_dir / f"{ts}_chat_payload.json"
        raw_path = self.debug_dir / f"{ts}_raw_response.json"

        payload = {
            "model_name": self.config.model_name,
            "base_url": self.config.base_url,
            "prompt_snapshot": prompt_snapshot,
        }

        prompt_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        raw_path.write_text(
            json.dumps(raw_response, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )