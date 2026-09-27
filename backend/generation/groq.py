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
    ) -> None:
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key and client is None:
            raise GenerationError("GROQ_API_KEY is not configured.")
        self._api_key = key
        self._model: str = model or os.getenv("GROQ_MODEL") or "openai/gpt-oss-20b"
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
        try:
            response = self._client.post(
                "/chat/completions",
                json={
                    "model": self._model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Answer only from the supplied context.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            if exc.response.status_code == 404:
                raise GenerationError(
                    f"Groq model '{self._model}' was not found or is unavailable. "
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
