"""Unit tests for query-time retrieval."""

from collections.abc import Sequence

import pytest

from backend.retrieval import RetrievalError, RetrievalService


class FakeEmbedder:
    """Capture the query and return a deterministic vector."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def embed(self, text: str) -> Sequence[float]:
        self.queries.append(text)
        return (0.1, 0.2)


class FakeRetriever:
    """Return controlled ranked chunks and scores."""

    def __init__(self) -> None:
        self.embedding: Sequence[float] | None = None
        self.top_k: int | None = None

    def query(
        self, query_embedding: Sequence[float], similarity_top_k: int
    ) -> tuple[Sequence[str], Sequence[float]]:
        self.embedding = query_embedding
        self.top_k = similarity_top_k
        return ("relevant context", "less relevant context"), (0.91, 0.42)


def test_retrieve_embeds_query_and_returns_aligned_scores() -> None:
    embedder = FakeEmbedder()
    store = FakeRetriever()

    result = RetrievalService(embedder, store).retrieve("What is AOA?")

    assert embedder.queries == ["What is AOA?"]
    assert store.embedding == (0.1, 0.2)
    assert store.top_k == 5
    assert result.retrieved_chunks == ("relevant context", "less relevant context")
    assert result.retrieval_scores == (0.91, 0.42)


def test_retrieve_honors_custom_top_k() -> None:
    store = FakeRetriever()

    RetrievalService(FakeEmbedder(), store, top_k=2).retrieve("question")

    assert store.top_k == 2


@pytest.mark.parametrize("query", ["", " \n"])
def test_empty_query_is_rejected(query: str) -> None:
    with pytest.raises(RetrievalError, match="Query must contain"):
        RetrievalService(FakeEmbedder(), FakeRetriever()).retrieve(query)


def test_mismatched_store_results_are_rejected() -> None:
    class MismatchedRetriever:
        def query(
            self, query_embedding: Sequence[float], similarity_top_k: int
        ) -> tuple[Sequence[str], Sequence[float]]:
            return ("chunk",), ()

    with pytest.raises(RetrievalError, match="mismatched"):
        RetrievalService(FakeEmbedder(), MismatchedRetriever()).retrieve("question")
