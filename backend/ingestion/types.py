"""Typed contracts and domain objects for document ingestion."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


class IngestionError(Exception):
    """Base error for expected document-ingestion failures."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when a document extension has no configured loader."""


@dataclass(frozen=True)
class DocumentPage:
    """Text extracted from one logical document page."""

    text: str
    page_number: int


@dataclass(frozen=True)
class TextChunk:
    """A chunk of source text and its document provenance."""

    text: str
    source: str
    page_number: int
    chunk_id: str


@dataclass(frozen=True)
class EmbeddedChunk:
    """A chunk together with its embedding and persisted metadata."""

    text: str
    embedding: tuple[float, ...]
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class IngestionResult:
    """Summary returned after all chunks have been embedded and stored."""

    source: str
    chunk_count: int
    chunks: tuple[EmbeddedChunk, ...]


class DocumentLoader(Protocol):
    """Protocol implemented by document parsers."""

    def load(self, path: Path) -> Sequence[DocumentPage]:
        """Parse ``path`` into ordered pages."""


class Embedder(Protocol):
    """Protocol implemented by embedding providers."""

    def embed(self, text: str) -> Sequence[float]:
        """Return an embedding for ``text``."""


class VectorStore(Protocol):
    """Protocol implemented by vector database adapters."""

    def add(self, chunks: Sequence[EmbeddedChunk]) -> None:
        """Persist embedded chunks, replacing entries with matching IDs."""
