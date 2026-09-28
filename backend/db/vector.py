"""LlamaIndex PostgreSQL vector-store helpers."""

from collections.abc import Sequence
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session


def metadata_filters(user_id: str, document_ids: Sequence[str] = ()) -> dict[str, Any]:
    """Build mandatory ownership/document metadata filters for vector queries."""
    filters: dict[str, Any] = {"user_id": user_id}
    if document_ids:
        filters["document_id"] = {"$in": list(document_ids)}
    return filters


def validate_embedding_dimension(actual: int, expected: int) -> None:
    """Fail fast when the vector table dimension differs from configuration."""
    if actual != expected:
        raise ValueError(
            f"Vector table dimension {actual} does not match EMBED_DIM={expected}."
        )


def vector_store_kwargs(database_url: str, embed_dim: int) -> dict[str, Any]:
    """Return safe PGVectorStore construction arguments for LlamaIndex."""
    return {
        "connection_string": database_url,
        "embed_dim": embed_dim,
        "table_name": "aoa_embeddings",
        "schema_name": "public",
    }


def create_vector_store(database_url: str, embed_dim: int) -> Any:
    """Create the configured LlamaIndex PostgreSQL vector store."""
    from llama_index.vector_stores.postgres import PGVectorStore

    return PGVectorStore(**vector_store_kwargs(database_url, embed_dim), use_jsonb=True)


def delete_document_vectors(session: Session, document_id: UUID) -> int:
    """Delete vector rows for a document in the caller's transaction."""
    result = session.execute(
        text(
            "DELETE FROM aoa_embeddings "
            "WHERE metadata_->>'document_id' = :document_id"
        ),
        {"document_id": str(document_id)},
    )
    return int(getattr(result, "rowcount", 0) or 0)
