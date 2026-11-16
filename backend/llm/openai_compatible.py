"""HTTP client for OpenAI Chat Completions-compatible model endpoints."""

from collections.abc import Mapping, Sequence
from typing import Any

import httpx


class ChatCompletionError(RuntimeError):
    """Raised when a configured chat-completion endpoint cannot answer."""


class ChatCompletionClient:
    """Call any model endpoint implementing OpenAI Chat Completions."""

    def __init__(
        self,
        *,
        model: str,
        api_base_url: str,
        api_key: str | None = None,
        timeout: float = 60.0,
        client: Any | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("A chat model name is required.")
        if not api_base_url.strip():
            raise ValueError("A chat model API base URL is required.")
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._model = model
        self._client = client or httpx.Client(
            base_url=f"{api_base_url.rstrip('/')}/",
            headers=headers,
            timeout=timeout,
        )

    def complete(
        self,
        messages: Sequence[Mapping[str, str]],
        *,
        temperature: float = 0.0,
    ) -> str:
        """Send messages to the configured model and return its text response."""
        try:
            response = self._client.post(
                "chat/completions",
                json={
                    "model": self._model,
                    "messages": list(messages),
                    "temperature": temperature,
                },
            )
            response.raise_for_status()
            payload = response.json()
            content = payload["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:500]
            raise ChatCompletionError(
                f"Model endpoint returned HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ChatCompletionError(f"Model endpoint request failed: {exc}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise ChatCompletionError(
                f"Model endpoint returned an invalid response: {exc}"
            ) from exc
        if not isinstance(content, str) or not content.strip():
            raise ChatCompletionError("Model endpoint returned empty response text.")
        return content.strip()
