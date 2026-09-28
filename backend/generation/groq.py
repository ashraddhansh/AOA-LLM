"""Groq API adapter for grounded answer generation."""

import os
from typing import Any

import httpx

from .types import GenerationError


class GroqGenerator:
    """Generate grounded answers through Groq's chat-completions API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
        fallback_models: tuple[str, ...] = (),
    ) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key and client is None:
            raise GenerationError("GROQ_API_KEY is not configured.")
        self._api_key = key
        self._model: str = model or os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b"
        self._fallback_models = fallback_models
        self._client = client or httpx.Client(
            base_url="https://api.groq.com/openai/v1",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            timeout=60.0,
        )

    def generate(self, prompt: str) -> str:
        """Send a grounded prompt to Groq and return the assistant text."""
        models = (self._model, *self._fallback_models)
        last_error: GenerationError | None = None
        for model in models:
            try:
                return self._generate_with_model(prompt, model)
            except GenerationError as exc:
                last_error = exc
                if not self._is_retryable_model_error(exc):
                    raise
        discovered = self._discover_models()
        for model in discovered:
            if model in models:
                continue
            try:
                return self._generate_with_model(prompt, model)
            except GenerationError as exc:
                last_error = exc
        raise last_error or GenerationError("Groq returned no answer.")

    def _generate_with_model(self, prompt: str, model: str) -> str:
        """Generate with one selected model."""
        try:
            response = self._client.post(
                "/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "Answer only from the supplied context. "
                                "Return the answer as clear Markdown, not HTML."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                    "tool_choice": "none",
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            if "tool_use_failed" in detail or "called a tool" in detail.lower():
                raise GenerationError(
                    "Groq attempted an unsupported tool call while generating text. "
                    "Tool calling is disabled for this RAG request. "
                    f"Groq response: {detail}"
                ) from exc
            if exc.response.status_code == 404:
                raise GenerationError(
                    f"Groq model '{model}' was not found or is unavailable. "
                    "Set GROQ_MODEL to an active chat model from Groq's model list. "
                    f"Groq response: {detail}"
                ) from exc
            raise GenerationError(
                f"Groq request failed with HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise GenerationError(f"Groq request failed: {exc}") from exc
        except (TypeError, ValueError, KeyError) as exc:
            raise GenerationError(f"Invalid Groq response: {exc}") from exc

        try:
            answer = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError("Groq returned no answer text.") from exc
        if not isinstance(answer, str) or not answer.strip():
            raise GenerationError("Groq returned empty answer text.")
        return answer

    @staticmethod
    def _is_retryable_model_error(error: GenerationError) -> bool:
        """Identify model availability/tool failures safe for fallback."""
        message = str(error)
        return (
            "not found or is unavailable" in message
            or "unsupported tool call" in message
        )

    def _discover_models(self) -> tuple[str, ...]:
        """Return accessible text-capable model IDs from Groq's model endpoint."""
        try:
            response = self._client.get("/models")
            response.raise_for_status()
            payload = response.json()
            models = payload.get("data", [])
        except (httpx.HTTPError, TypeError, ValueError, AttributeError):
            return ()
        if not isinstance(models, list):
            return ()
        preferred = (
            "llama",
            "qwen",
            "mixtral",
            "gemma",
            "openai/gpt-oss",
        )
        return tuple(
            item["id"]
            for item in models
            if isinstance(item, dict)
            and isinstance(item.get("id"), str)
            and any(token in item["id"].lower() for token in preferred)
        )
