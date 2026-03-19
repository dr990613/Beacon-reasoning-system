# src/beacon_system/llm/client.py
# -*- coding: utf-8 -*-

"""
LLM Client (OpenAI-compatible)

- Injected dependency for planning / generator / revise.
- Business modules MUST NOT read env; they only receive an LLMClient instance.
- Talks to OpenAI-compatible servers:
  - POST {base_url}/chat/completions
  - POST {base_url}/completions

Current target backend:
- OpenRouter + Qwen, e.g.
  - base_url = https://openrouter.ai/api/v1
  - model_name = qwen/qwen3-32b

Returns plain text content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import json
import os
import time

import requests

from .config import ModelConfig


class LLMError(RuntimeError):
    """Raised when LLM request/response handling fails."""
    pass


def _join_url(base_url: str, path: str) -> str:
    b = (base_url or "").rstrip("/")
    p = (path or "").lstrip("/")
    return f"{b}/{p}"


def _headers(api_key: str, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    h = {
        "Content-Type": "application/json",
    }
    if api_key and api_key.strip():
        h["Authorization"] = f"Bearer {api_key.strip()}"

    if extra_headers:
        for k, v in extra_headers.items():
            if v is not None and str(v).strip():
                h[str(k)] = str(v)

    return h


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        try:
            return repr(obj)
        except Exception:
            return "<unserializable>"


def _debug_dir() -> str:
    """
    Optional debug output dir.
    Kept as env-based for local debugging convenience only.
    """
    return os.environ.get("LLM_DEBUG_DIR", "outputs/llm_debug")


def _save_debug_json(tag: str, payload: Any) -> None:
    try:
        os.makedirs(_debug_dir(), exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        path = os.path.join(_debug_dir(), f"{ts}_{tag}.json")
        with open(path, "w", encoding="utf-8") as f:
            f.write(_safe_json_dumps(payload))
    except Exception:
        pass


def _extract_text_from_content(content: Any) -> str:
    """
    Compatible extraction for OpenAI/OpenRouter-like content payloads.

    Supported:
    - str
    - list[{"type":"text","text":"..."}]
    - list[mixed]
    - dict with text/content/output_text
    """
    if content is None:
        return ""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
                if text:
                    parts.append(text)
                continue

            if not isinstance(item, dict):
                continue

            if item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]).strip())
                continue

            for key in ("text", "content", "output_text"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    parts.append(value.strip())
                    break

        return "\n".join([p for p in parts if p]).strip()

    if isinstance(content, dict):
        for key in ("text", "content", "output_text"):
            value = content.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

    return ""


def _extract_chat_text(resp_json: Dict[str, Any]) -> str:
    """
    Compatible extraction for OpenAI/OpenRouter-like chat responses.

    Common formats:
    - choices[0].message.content -> str
    - choices[0].message.content -> list[{"type":"text","text":"..."}]
    - choices[0].message.reasoning / text / output_text
    - choices[0].text
    - top-level output_text / text
    - top-level output -> list[{content:[...]}]
    """
    try:
        choices = resp_json.get("choices") or []
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue

                msg = choice.get("message") or {}
                if isinstance(msg, dict):
                    text = _extract_text_from_content(msg.get("content"))
                    if text:
                        return text

                    for key in ("reasoning", "text", "output_text"):
                        value = msg.get(key)
                        if isinstance(value, str) and value.strip():
                            return value.strip()

                text = _extract_text_from_content(choice.get("content"))
                if text:
                    return text

                for key in ("text", "output_text"):
                    value = choice.get(key)
                    if isinstance(value, str) and value.strip():
                        return value.strip()

        for key in ("output_text", "text"):
            value = resp_json.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        output = resp_json.get("output")
        if isinstance(output, list):
            parts: List[str] = []
            for item in output:
                if not isinstance(item, dict):
                    continue
                text = _extract_text_from_content(item.get("content"))
                if text:
                    parts.append(text)
                else:
                    for key in ("text", "output_text"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            parts.append(value.strip())
                            break
            if parts:
                return "\n".join(parts).strip()

        return ""
    except Exception:
        return ""


def _extract_completion_text(resp_json: Dict[str, Any]) -> str:
    """
    OpenAI completion-like formats:
    - choices[0].text
    - choices[0].message.content
    - top-level output_text / text
    """
    try:
        choices = resp_json.get("choices") or []
        if isinstance(choices, list):
            for choice in choices:
                if not isinstance(choice, dict):
                    continue

                text = choice.get("text")
                if text is not None and str(text).strip():
                    return str(text).strip()

                msg = choice.get("message") or {}
                if isinstance(msg, dict):
                    text2 = _extract_text_from_content(msg.get("content"))
                    if text2:
                        return text2

        for key in ("output_text", "text"):
            value = resp_json.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        return ""
    except Exception:
        return ""


def _extract_error_text(resp: requests.Response) -> str:
    try:
        data = resp.json()
        err = data.get("error")
        if isinstance(err, dict):
            msg = err.get("message") or err.get("type") or json.dumps(err, ensure_ascii=False)
            return str(msg)
        if err is not None:
            return str(err)
        return resp.text[:500]
    except Exception:
        return resp.text[:500]


@dataclass
class LLMClient:
    """
    Minimal OpenAI-compatible client.

    Notes:
    - No env access except optional debug output path helper.
    - Includes light retry for empty responses / transient request failures.
    - Shared by planning / generator / revise.
    """
    cfg: ModelConfig

    def _print_enabled(self) -> bool:
        """
        Controlled debug print switch.

        Safe default:
        - off unless cfg explicitly exposes `print_io` / `verbose` = True
        """
        try:
            return bool(
                getattr(self.cfg, "print_io", False)
                or getattr(self.cfg, "verbose", False)
            )
        except Exception:
            return False

    def _print(self, message: str) -> None:
        if self._print_enabled():
            print(f"[LLMClient] {message}")

    def _build_extra_headers(self) -> Dict[str, str]:
        """
        Optional headers for OpenRouter attribution/ranking.
        They are harmless for most OpenAI-compatible providers.

        These fields are optional on ModelConfig.
        If your ModelConfig does not define them, getattr(..., None) keeps it safe.
        """
        extra: Dict[str, str] = {}

        referer = getattr(self.cfg, "site_url", None)
        app_name = getattr(self.cfg, "app_name", None)

        if referer:
            extra["HTTP-Referer"] = str(referer)
        if app_name:
            extra["X-Title"] = str(app_name)

        return extra

    def _max_retries(self) -> int:
        value = getattr(self.cfg, "max_retries", None)
        if value is None:
            return 2
        try:
            return max(0, int(value))
        except Exception:
            return 2

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = _join_url(self.cfg.base_url, path)
        self._print(f"POST {url}")
        self._print(f"model={payload.get('model')} timeout={self.cfg.timeout_s}s")

        try:
            r = requests.post(
                url,
                headers=_headers(self.cfg.api_key, self._build_extra_headers()),
                data=json.dumps(payload, ensure_ascii=False),
                timeout=self.cfg.timeout_s,
            )
        except requests.RequestException as e:
            self._print(f"request exception: {e}")
            raise LLMError(f"LLM request failed: {e}") from e

        if r.status_code >= 400:
            err_text = _extract_error_text(r)
            self._print(f"http error: status={r.status_code} error={err_text}")
            _save_debug_json("http_error_payload", payload)
            try:
                _save_debug_json("http_error_response", r.json())
            except Exception:
                _save_debug_json("http_error_response_text", {"text": r.text[:2000]})
            raise LLMError(f"LLM request failed: {r.status_code} {err_text}")

        try:
            resp_json = r.json()
            _save_debug_json("raw_response", resp_json)
            self._print("response received: json ok")
            return resp_json
        except Exception as e:
            self._print("response parse failed: non-json")
            _save_debug_json("non_json_payload", payload)
            _save_debug_json("non_json_response_text", {"text": r.text[:5000]})
            raise LLMError(f"LLM returned non-JSON response: {r.text[:500]}") from e

    def chat(
        self,
        messages: List[Dict[str, str]],
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model or self.cfg.model_name,
            "messages": messages,
        }
        payload.update(dict(self.cfg.params or {}))
        payload.update(kwargs)

        retries = self._max_retries()
        last_empty_resp: Optional[Dict[str, Any]] = None
        last_error: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                self._print(f"chat attempt={attempt + 1}/{retries + 1}")
                _save_debug_json("chat_payload", payload)
                resp_json = self._post("/chat/completions", payload)

                text = _extract_chat_text(resp_json)
                if not text:
                    text = _extract_completion_text(resp_json)

                if text:
                    self._print(f"chat success: chars={len(text)}")
                    return text

                last_empty_resp = resp_json
                self._print("chat returned empty text")
                _save_debug_json("empty_chat_response", resp_json)

                if attempt < retries:
                    time.sleep(1.0 + attempt)
                    continue

            except Exception as e:
                last_error = e
                self._print(f"chat exception on attempt {attempt + 1}: {e}")
                _save_debug_json("chat_exception", {"error": repr(e), "attempt": attempt})

                if attempt < retries:
                    time.sleep(1.0 + attempt)
                    continue
                break

        if last_error is not None:
            raise LLMError(f"LLM chat failed after retries: {last_error}")

        if last_empty_resp is not None:
            raise LLMError(
                "LLM chat returned empty content. "
                f"Debug response saved under {_debug_dir()}."
            )

        raise LLMError("LLM chat failed for unknown reason.")

    def complete(
        self,
        prompt: str,
        *,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model or self.cfg.model_name,
            "prompt": prompt,
        }
        payload.update(dict(self.cfg.params or {}))
        payload.update(kwargs)

        retries = self._max_retries()
        last_empty_resp: Optional[Dict[str, Any]] = None
        last_error: Optional[Exception] = None

        for attempt in range(retries + 1):
            try:
                self._print(f"completion attempt={attempt + 1}/{retries + 1}")
                _save_debug_json("completion_payload", payload)
                resp_json = self._post("/completions", payload)

                text = _extract_completion_text(resp_json)
                if not text:
                    text = _extract_chat_text(resp_json)

                if text:
                    self._print(f"completion success: chars={len(text)}")
                    return text

                last_empty_resp = resp_json
                self._print("completion returned empty text")
                _save_debug_json("empty_completion_response", resp_json)

                if attempt < retries:
                    time.sleep(1.0 + attempt)
                    continue

            except Exception as e:
                last_error = e
                self._print(f"completion exception on attempt {attempt + 1}: {e}")
                _save_debug_json("completion_exception", {"error": repr(e), "attempt": attempt})

                if attempt < retries:
                    time.sleep(1.0 + attempt)
                    continue
                break

        if last_error is not None:
            try:
                self._print("completion failed; trying chat fallback")
                text = self.chat(
                    [{"role": "user", "content": prompt}],
                    model=model,
                    **kwargs,
                )
                if text:
                    return text
            except Exception as fallback_error:
                raise LLMError(
                    f"LLM completion failed after retries: {last_error}; "
                    f"chat fallback failed: {fallback_error}"
                ) from fallback_error

            raise LLMError(f"LLM completion failed after retries: {last_error}")

        if last_empty_resp is not None:
            try:
                self._print("completion empty; trying chat fallback")
                text = self.chat(
                    [{"role": "user", "content": prompt}],
                    model=model,
                    **kwargs,
                )
                if text:
                    return text
            except Exception as fallback_error:
                raise LLMError(
                    "LLM completion returned empty content and chat fallback failed: "
                    f"{fallback_error}"
                ) from fallback_error

            raise LLMError(
                "LLM completion returned empty content. "
                f"Debug response saved under {_debug_dir()}."
            )

        raise LLMError("LLM completion failed for unknown reason.")

    def generate_text(
        self,
        *,
        system: Optional[str] = None,
        user: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """
        Unified text-generation entry for agent modules.

        Preferred usage:
        - planning / generator / revise call this method
        - pass `messages` directly, or pass `system` + `user`

        Returns:
            plain text content
        """
        if messages is not None:
            if not isinstance(messages, list) or not messages:
                raise LLMError("generate_text(messages=...) requires a non-empty message list.")
            return self.chat(messages, model=model, **kwargs)

        built_messages: List[Dict[str, str]] = []
        if system and str(system).strip():
            built_messages.append({"role": "system", "content": str(system)})
        if user and str(user).strip():
            built_messages.append({"role": "user", "content": str(user)})

        if not built_messages:
            raise LLMError("generate_text requires either messages or user/system input.")

        return self.chat(built_messages, model=model, **kwargs)