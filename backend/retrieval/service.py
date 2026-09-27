"""Application service for query-time similarity retrieval."""

from .types import QueryEmbedder, RetrievalError, RetrievalResult, VectorRetriever


class RetrievalService:
    """Embed a query and retrieve its most similar document chunks."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        vector_store: VectorRetriever,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        self._embedder = embedder
        self._vector_store = vector_store
        self._top_k = top_k

    def retrieve(
        self, query: str, filters: dict[str, str | int] | None = None
    ) -> RetrievalResult:
        """Return up to ``top_k`` chunks and their aligned similarity scores."""
        if not query.strip():
            raise RetrievalError(
                "Query must contain at least one non-whitespace character."
            )

        query_embedding = tuple(self._embedder.embed(query))
        if not query_embedding:
            raise RetrievalError("Query embedder returned an empty vector.")

        try:
            chunks, scores = self._vector_store.query(
                query_embedding, self._top_k, filters
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise RetrievalError(f"Similarity search failed: {exc}") from exc

        if len(chunks) != len(scores):
            raise RetrievalError(
                "Vector store returned mismatched chunks and retrieval scores."
            )
        return RetrievalResult(
            query=query,
            retrieved_chunks=tuple(chunks),
            retrieval_scores=tuple(float(score) for score in scores),
        )
