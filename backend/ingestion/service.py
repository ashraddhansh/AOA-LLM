"""The document ingestion application service."""

from collections.abc import Callable, Sequence
from pathlib import Path

from .llama_index_adapter import LlamaIndexChunker, LlamaIndexDocumentLoader
from .types import (
    DocumentLoader,
    DocumentPage,
    EmbeddedChunk,
    Embedder,
    IngestionError,
    IngestionResult,
    TextChunk,
    VectorStore,
)


class IngestionService:
    """Parse, chunk, embed, and persist one uploaded document."""

    def __init__(
        self,
        loader: DocumentLoader | None,
        embedder: Embedder,
        vector_store: VectorStore,
        chunk_size: int = 512,
        overlap: int = 64,
        chunker: (
            Callable[[Sequence[DocumentPage], Path, int, int], tuple[TextChunk, ...]]
            | None
        ) = None,
    ) -> None:
        self._loader = loader or LlamaIndexDocumentLoader()
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._chunker = chunker or LlamaIndexChunker(chunk_size, overlap)

    def ingest(
        self, path: Path, metadata: dict[str, str | int] | None = None
    ) -> IngestionResult:
        """Ingest ``path`` or raise a clear error without partial persistence."""
        pages = tuple(self._loader.load(path))
        chunks = self._chunker(pages, path, self._chunk_size, self._overlap)
        if not chunks:
            raise IngestionError(f"Document '{path}' contains no extractable text.")

        embedded: list[EmbeddedChunk] = []
        for chunk in chunks:
            vector = tuple(self._embedder.embed(chunk.text))
            if not vector:
                raise IngestionError(
                    f"Embedder returned an empty vector for {chunk.chunk_id}."
                )
            chunk_metadata: dict[str, str | int] = {
                "source": chunk.source,
                "page": chunk.page_number,
                "chunk_id": chunk.chunk_id,
            }
            if metadata:
                chunk_metadata.update(metadata)
                if "chat_id" in metadata:
                    chunk_metadata["chunk_id"] = (
                        f"{metadata['chat_id']}:{chunk.chunk_id}"
                    )
            embedded.append(
                EmbeddedChunk(
                    text=chunk.text,
                    embedding=vector,
                    metadata=chunk_metadata,
                )
            )
        self._vector_store.add(embedded)
        return IngestionResult(
            source=str(path),
            chunk_count=len(embedded),
            chunks=tuple(embedded),
        )
