"""Small HTTP clients for local Ollama and hosted OpenAI-compatible models."""

from __future__ import annotations

import time
from typing import Any, Protocol

import httpx

from app.core.config import Settings
from app.generation.errors import (
    LLMConfigError,
    LLMError,
    LLMResponseError,
    LLMTimeoutError,
)


class LLMClient(Protocol):
    def generate(self, system: str, user: str) -> str: ...


class _HttpLLMClient:
    def __init__(self, *, base_url: str, model: str, timeout: float, retries: int,
                 max_output_tokens: int, api_key: str | None = None,
                 transport: httpx.BaseTransport | None = None) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._retries = retries
        self._max_tokens = max_output_tokens
        self._api_key = api_key
        self._transport = transport

    def _request(self, payload: dict[str, Any], *, ollama: bool) -> str:
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        retryable = (httpx.TimeoutException, httpx.TransportError)
        for attempt in range(self._retries + 1):
            try:
                # Localhost traffic must bypass corporate/system proxies; hosted
                # endpoints keep httpx's normal environment proxy behavior.
                with httpx.Client(
                    timeout=self._timeout,
                    transport=self._transport,
                    trust_env=not ollama,
                ) as client:
                    response = client.post(
                        f"{self._base_url}/api/chat" if ollama else f"{self._base_url}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                if (response.status_code == 429 or response.status_code >= 500) and attempt < self._retries:
                    time.sleep(min(0.1 * (2**attempt), 1.0))
                    continue
                response.raise_for_status()
                try:
                    data = response.json()
                    answer = data["message"]["content"] if ollama else data["choices"][0]["message"]["content"]
                except (ValueError, KeyError, IndexError, TypeError) as exc:
                    raise LLMResponseError() from exc
                if not isinstance(answer, str) or not answer.strip():
                    raise LLMResponseError()
                return answer.strip()
            except retryable as exc:
                if attempt < self._retries:
                    time.sleep(min(0.1 * (2**attempt), 1.0))
                    continue
                if isinstance(exc, httpx.TimeoutException):
                    raise LLMTimeoutError() from exc
                raise LLMError() from exc
            except LLMResponseError:
                raise
            except httpx.HTTPStatusError as exc:
                # Avoid passing provider response bodies or secrets into logs/errors.
                raise LLMError() from exc
        raise LLMError()


class OllamaLLMClient(_HttpLLMClient):
    def generate(self, system: str, user: str) -> str:
        return self._request({
            "model": self._model,
            "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"num_predict": self._max_tokens, "temperature": 0.1},
        }, ollama=True)


class HostedLLMClient(_HttpLLMClient):
    def generate(self, system: str, user: str) -> str:
        return self._request({
            "model": self._model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": self._max_tokens,
            "temperature": 0.1,
        }, ollama=False)


class FakeLLMClient:
    def __init__(self, answer: str = "fake answer") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def generate(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.answer


def build_llm_client(settings: Settings) -> LLMClient:
    if settings.llm_provider == "ollama":
        if not settings.ollama_model.strip():
            raise LLMConfigError("Ollama model is required.")
        return OllamaLLMClient(
            base_url=settings.ollama_base_url, model=settings.ollama_model,
            timeout=settings.llm_timeout_seconds, retries=settings.llm_max_retries,
            max_output_tokens=settings.llm_max_output_tokens,
        )
    if settings.llm_provider == "hosted":
        key = settings.hosted_llm_api_key.get_secret_value() if settings.hosted_llm_api_key else ""
        if not key.strip() or not settings.hosted_llm_model.strip():
            raise LLMConfigError("Hosted provider model and API key are required.")
        return HostedLLMClient(
            base_url=settings.hosted_llm_base_url, model=settings.hosted_llm_model,
            timeout=settings.llm_timeout_seconds, retries=settings.llm_max_retries,
            max_output_tokens=settings.llm_max_output_tokens, api_key=key,
        )
    raise LLMConfigError("Unsupported language model provider.")
