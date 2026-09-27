"""Gemini API adapter for grounded answer generation."""

import os
from typing import Any

from google import genai
from google.genai import errors

from .types import GenerationError


class GeminiGenerator:
    """Generate grounded answers using Google's Gemini API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: Any | None = None,
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key and client is None:
            raise GenerationError("GEMINI_API_KEY is not configured.")
        self._client = client or genai.Client(api_key=key)
        self._model: str = model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash"

    def generate(self, prompt: str) -> str:
        """Send the complete grounded prompt to Gemini and return raw text."""
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
        except errors.APIError as exc:
            if exc.code == 404:
                raise GenerationError(
                    f"Gemini model '{self._model}' was not found or is unavailable "
                    "for this API key. Set GEMINI_MODEL to a model returned by "
                    "the Gemini models list."
                ) from exc
            raise GenerationError(f"Gemini request failed: {exc}") from exc
        except (OSError, RuntimeError, ValueError) as exc:
            raise GenerationError(f"Gemini request failed: {exc}") from exc
        answer = getattr(response, "text", None)
        if not isinstance(answer, str) or not answer.strip():
            raise GenerationError("Gemini returned no answer text.")
        return answer
