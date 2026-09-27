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
        model: str = "gemini-2.5-flash",
        client: Any | None = None,
    ) -> None:
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key and client is None:
            raise GenerationError("GEMINI_API_KEY is not configured.")
        self._client = client or genai.Client(api_key=key)
        self._model = model

    def generate(self, prompt: str) -> str:
        """Send the complete grounded prompt to Gemini and return raw text."""
        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
            )
        except (errors.APIError, OSError, RuntimeError, ValueError) as exc:
            raise GenerationError(f"Gemini request failed: {exc}") from exc
        answer = getattr(response, "text", None)
        if not isinstance(answer, str) or not answer.strip():
            raise GenerationError("Gemini returned no answer text.")
        return answer
