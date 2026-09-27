"""Typed contracts for query-time retrieval."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


class RetrievalError(Exception):
    """Raised when query embedding or vector retrieval cannot complete."""


@dataclass(frozen=True)
class RetrievalResult:
    """Retrieved context and scores aligned by rank."""

    query: str
    retrieved_chunks: tuple[str, ...]
    retrieval_scores: tuple[float, ...]


class QueryEmbedder(Protocol):
    """Protocol for models that embed user queries."""

    def embed(self, text: str) -> Sequence[float]:
        """Return an embedding for the query text."""


class VectorRetriever(Protocol):
    """Protocol for vector stores that support similarity search."""

    def query(
        self, query_embedding: Sequence[float], similarity_top_k: int
    ) -> tuple[Sequence[str], Sequence[float]]:
        """Return chunk text and similarity scores ordered by relevance."""
