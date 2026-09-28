"""SerpAPI web-search adapter."""

import os
from collections.abc import Mapping
from typing import Any

import httpx


class SearchError(Exception):
    """Raised when SerpAPI cannot return usable search results."""


class SerpApiSearch:
    """Search the live web through SerpAPI's Google engine."""

    def __init__(self, api_key: str | None = None, client: Any | None = None) -> None:
        self._api_key = (
            api_key or os.getenv("SERPAPI_API_KEY") or os.getenv("WEB_SEARCH_API_KEY")
        )
        if not self._api_key and client is None:
            raise SearchError("SERPAPI_API_KEY is not configured.")
        self._client = client or httpx.Client(timeout=30.0)

    def search(self, query: str, limit: int = 5) -> tuple[str, ...]:
        """Return formatted organic search results for ``query``."""
        if not query.strip():
            raise SearchError("Search query cannot be empty.")
        try:
            response = self._client.get(
                "https://serpapi.com/search",
                params={
                    "engine": "google",
                    "q": query,
                    "api_key": self._api_key,
                    "num": limit,
                },
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as exc:
            raise SearchError(
                f"SerpAPI request failed with HTTP {exc.response.status_code}: "
                f"{exc.response.text[:500]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SearchError(f"SerpAPI request failed: {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise SearchError(f"Invalid SerpAPI response: {exc}") from exc

        results = payload.get("organic_results", [])
        if not isinstance(results, list):
            raise SearchError("SerpAPI returned invalid organic results.")
        formatted = tuple(
            _format_result(item)
            for item in results[:limit]
            if isinstance(item, Mapping)
        )
        formatted = tuple(item for item in formatted if item)
        if not formatted:
            raise SearchError("SerpAPI returned no usable search results.")
        return formatted


def _format_result(result: Mapping[str, Any]) -> str:
    """Format one SerpAPI result as grounded generation context."""
    title = result.get("title")
    snippet = result.get("snippet")
    link = result.get("link")
    if not isinstance(title, str) or not title.strip():
        return ""
    if not isinstance(snippet, str) or not snippet.strip():
        return ""
    return (
        f"Title: {title.strip()}\nSnippet: {snippet.strip()}\nURL: {link or 'unknown'}"
    )
