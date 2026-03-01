# src/beacon_system/llm/client.py
# -*- coding: utf-8 -*-

"""
LLM Client (OpenAI-compatible)

- Injected dependency for generator (and only generator should call it).
- Business modules MUST NOT read env; they only receive an LLMClient instance.
- Talks to liteLLM/vLLM in OpenAI-compatible mode:
  - POST {base_url}/chat/completions
  - POST {base_url}/completions

Returns plain text content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import requests

from .config import ModelConfig


class LLMError(RuntimeError):
    pass


def _join_url(base_url: str, path: str) -> str:
    b = (base_url or "").rstrip("/")
    p = (path or "").lstrip("/")
    return f"{b}/{p}"


def _headers(api_key: str) -> Dict[str, str]:
    h = {"Content-Type": "application/json"}
    # OpenAI-compatible servers commonly accept Authorization: Bearer
    if api_key and api_key.strip():
        h["Authorization"] = f"Bearer {api_key.strip()}"
    return h


def _extract_chat_text(resp_json: Dict[str, Any]) -> str:
    # OpenAI chat format: choices[0].message.content
    try:
        choices = resp_json.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        return (msg.get("content") or "").strip()
    except Exception:
        return ""


def _extract_completion_text(resp_json: Dict[str, Any]) -> str:
    # OpenAI completion format: choices[0].text
    try:
        choices = resp_json.get("choices") or []
        if not choices:
            return ""
        return (choices[0].get("text") or "").strip()
    except Exception:
        return ""


@dataclass
class LLMClient:
    """
    Minimal OpenAI-compatible client.

    NOTE:
    - No env access.
    - No caching / retry policy here (keep minimal). Retries can be added later in pipeline if needed.
    """
    cfg: ModelConfig

    def chat(self, messages: List[Dict[str, str]], *, model: Optional[str] = None, **kwargs: Any) -> str:
        url = _join_url(self.cfg.base_url, "/chat/completions")
        payload: Dict[str, Any] = {
            "model": model or self.cfg.model_name,
            "messages": messages,
        }
        payload.update(dict(self.cfg.params or {}))
        payload.update(kwargs)

        r = requests.post(
            url,
            headers=_headers(self.cfg.api_key),
            data=json.dumps(payload, ensure_ascii=False),
            timeout=self.cfg.timeout_s,
        )
        if r.status_code >= 400:
            raise LLMError(f"LLM chat failed: {r.status_code} {r.text[:500]}")

        resp_json = r.json()
        text = _extract_chat_text(resp_json)
        if not text:
            # Some servers may respond with completion-like format; fallback:
            text = _extract_completion_text(resp_json)
        if not text:
            raise LLMError("LLM chat returned empty content.")
        return text

    def complete(self, prompt: str, *, model: Optional[str] = None, **kwargs: Any) -> str:
        url = _join_url(self.cfg.base_url, "/completions")
        payload: Dict[str, Any] = {
            "model": model or self.cfg.model_name,
            "prompt": prompt,
        }
        payload.update(dict(self.cfg.params or {}))
        payload.update(kwargs)

        r = requests.post(
            url,
            headers=_headers(self.cfg.api_key),
            data=json.dumps(payload, ensure_ascii=False),
            timeout=self.cfg.timeout_s,
        )
        if r.status_code >= 400:
            raise LLMError(f"LLM completion failed: {r.status_code} {r.text[:500]}")

        resp_json = r.json()
        text = _extract_completion_text(resp_json)
        if not text:
            # Some servers only support chat; fallback to chat with a single user message
            text = self.chat([{"role": "user", "content": prompt}], model=model, **kwargs)
        if not text:
            raise LLMError("LLM completion returned empty content.")
        return text