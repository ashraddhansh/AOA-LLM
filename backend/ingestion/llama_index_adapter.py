"""LlamaIndex-backed ingestion components.

The adapters keep the application's small typed contracts while delegating
parsing, token-aware splitting, embedding, and vector persistence to
LlamaIndex integrations.
"""

import hashlib
import math
from collections.abc import Sequence
from pathlib import Path

from llama_index.core import SimpleDirectoryReader
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode
from llama_index.core.vector_stores import (
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)
from llama_index.vector_stores.chroma import ChromaVectorStore

from .types import (
    Embedder,
    DocumentPage,
    EmbeddedChunk,
    IngestionError,
    TextChunk,
    UnsupportedFileTypeError,
)


class LlamaIndexDocumentLoader:
    """Load supported files through LlamaIndex's file readers."""

    _SUPPORTED_EXTENSIONS = frozenset(
        {".txt", ".md", ".markdown", ".csv", ".json", ".html", ".pdf", ".docx"}
    )

    def load(self, path: Path) -> Sequence[DocumentPage]:
        """Read ``path`` and return non-empty LlamaIndex documents as pages."""
        if not path.is_file():
            raise IngestionError(f"Document does not exist: {path}")
        if path.suffix.lower() not in self._SUPPORTED_EXTENSIONS:
            raise UnsupportedFileTypeError(
                f"Unsupported document type: {path.suffix or '<none>'}"
            )

        try:
            documents = SimpleDirectoryReader(
                input_files=[path], filename_as_id=True, raise_on_error=True
            ).load_data()
        except (OSError, ValueError) as exc:
            raise IngestionError(f"Unable to parse document '{path}': {exc}") from exc
        pages: list[DocumentPage] = []
        for document in documents:
            text = document.get_content().strip()
            if not text:
                continue
            page_label = document.metadata.get("page_label")
            try:
                page_number = (
                    int(page_label) if str(page_label).strip() else len(pages) + 1
                )
            except (TypeError, ValueError):
                page_number = len(pages) + 1
            pages.append(DocumentPage(text=text, page_number=page_number))
        return tuple(pages)


class LlamaIndexChunker:
    """Token-aware chunker backed by LlamaIndex ``SentenceSplitter``."""

    def __init__(self, chunk_size: int = 512, overlap: int = 64) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self._splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    def __call__(
        self,
        pages: Sequence[DocumentPage],
        source: Path,
        _chunk_size: int = 512,
        _overlap: int = 64,
    ) -> tuple[TextChunk, ...]:
        """Split pages and retain source/page provenance."""
        chunks: list[TextChunk] = []
        for page in pages:
            for text in self._splitter.split_text(page.text):
                if not text.strip():
                    continue
                chunks.append(
                    TextChunk(
                        text=text,
                        source=str(source),
                        page_number=page.page_number,
                        chunk_id=f"{source.name}:p{page.page_number}:c{len(chunks)}",
                    )
                )
        return tuple(chunks)


class LlamaIndexEmbedder:
    """Adapt a LlamaIndex embedding model to the application protocol."""

    def __init__(self, model: BaseEmbedding) -> None:
        self._model = model

    def embed(self, text: str) -> tuple[float, ...]:
        """Embed text using the configured LlamaIndex model."""
        return tuple(self._model.get_text_embedding(text))


class LocalHashEmbedding(BaseEmbedding):
    """Deterministic local embedding model for document and query vectors."""

    def __init__(self, dimensions: int = 256) -> None:
        super().__init__(model_name="local-hash")
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        self._dimensions = dimensions

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed(text)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._embed(query)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self._dimensions
        for token in text.lower().split():
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            vector[int.from_bytes(digest, "big") % self._dimensions] += 1.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


class LlamaIndexChromaStore:
    """Persist application chunks in a LlamaIndex Chroma vector store."""

    def __init__(self, store: ChromaVectorStore) -> None:
        self._store = store

    def add(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Atomically add embedded chunks as LlamaIndex ``TextNode`` objects."""
        nodes: list[BaseNode] = [
            TextNode(
                id_=str(chunk.metadata["chunk_id"]),
                text=chunk.text,
                embedding=list(chunk.embedding),
                metadata=dict(chunk.metadata),
            )
            for chunk in chunks
        ]
        if nodes:
            self._store.add(nodes)

    def query(
        self,
        query_embedding: Sequence[float],
        similarity_top_k: int,
        filters: dict[str, str | int] | None = None,
    ) -> tuple[Sequence[str], Sequence[float]]:
        """Run Chroma similarity search and return text with aligned scores."""
        if similarity_top_k <= 0:
            raise ValueError("similarity_top_k must be greater than zero")
        metadata_filters = None
        if filters:
            metadata_filters = MetadataFilters(
                filters=[
                    MetadataFilter(key=key, value=value)
                    for key, value in filters.items()
                ]
            )
        result = self._store.query(
            VectorStoreQuery(
                query_embedding=list(query_embedding),
                similarity_top_k=similarity_top_k,
                filters=metadata_filters,
            )
        )
        nodes = result.nodes or ()
        similarities = result.similarities or ()
        return (
            tuple(node.get_content() for node in nodes),
            tuple(float(score) for score in similarities),
        )

    @classmethod
    def from_persist_dir(
        cls, persist_dir: Path, collection_name: str = "aoa_documents_local_v2"
    ) -> "LlamaIndexChromaStore":
        """Create a persistent local Chroma-backed LlamaIndex store.

        The collection name is versioned because Chroma collections require a
        single embedding dimension. Changing embedding providers or models
        requires a new collection rather than mixing incompatible vectors.
        """
        try:
            import chromadb
        except ImportError as exc:  # pragma: no cover - dependency is packaged
            raise RuntimeError(
                "Chroma persistence requires the 'chromadb' package"
            ) from exc
        client = chromadb.PersistentClient(path=str(persist_dir))
        collection = client.get_or_create_collection(collection_name)
        return cls(ChromaVectorStore(chroma_collection=collection))


def migrate_legacy_chroma_collection(
    persist_dir: Path,
    source_collection_name: str,
    target_collection_name: str,
    embedder: Embedder,
) -> int:
    """Re-embed legacy Chroma text into the active collection, retaining metadata.

    Chroma collections cannot mix embedding dimensions. This migration reads
    stored text from the previous collection and writes it with the current
    embedder, preserving fields such as ``chat_id`` used to scope retrieval.
    Repeated runs skip chunk IDs already present in the target collection.
    """
    if source_collection_name == target_collection_name:
        return 0

    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    collections = client.list_collections()
    collection_names = {
        item if isinstance(item, str) else item.name for item in collections
    }
    if source_collection_name not in collection_names:
        return 0

    source = client.get_collection(source_collection_name)
    target = client.get_or_create_collection(target_collection_name)
    records = source.get(include=["documents", "metadatas"])
    ids = records.get("ids", [])
    documents = records.get("documents", [])
    metadatas = records.get("metadatas", [])
    existing_ids = set(target.get(include=[]).get("ids", []))
    migrated: list[EmbeddedChunk] = []

    for node_id, text, metadata in zip(ids, documents, metadatas, strict=True):
        if not text or not text.strip():
            continue
        chunk_id = f"legacy:{node_id}"
        if chunk_id in existing_ids:
            continue
        chunk_metadata: dict[str, str | int] = {
            key: value
            for key, value in (metadata or {}).items()
            if isinstance(value, (str, int))
        }
        chunk_metadata["chunk_id"] = chunk_id
        chunk_metadata.setdefault("source", "legacy-document")
        migrated.append(
            EmbeddedChunk(
                text=text,
                embedding=tuple(embedder.embed(text)),
                metadata=chunk_metadata,
            )
        )

    target_store = LlamaIndexChromaStore(
        ChromaVectorStore(chroma_collection=target)
    )
    target_store.add(migrated)
    return len(migrated)
