"""Unit tests for the document ingestion layer."""

from pathlib import Path

import pytest

from backend.ingestion import (
    DocumentPage,
    EmbeddedChunk,
    IngestionService,
    LlamaIndexEmbedder,
)
from backend.ingestion.llama_index_adapter import (
    LlamaIndexChunker,
    LlamaIndexDocumentLoader,
)
from backend.ingestion.types import IngestionError, UnsupportedFileTypeError


class FakeVectorStore:
    """Capture persisted chunks without requiring a Chroma server."""

    def __init__(self) -> None:
        self.chunks: list[EmbeddedChunk] = []

    def add(self, chunks: list[EmbeddedChunk]) -> None:
        self.chunks.extend(chunks)


class FakeLoader:
    """Return pages supplied by a test."""

    def __init__(self, pages: tuple[DocumentPage, ...]) -> None:
        self.pages = pages

    def load(self, path: Path) -> tuple[DocumentPage, ...]:
        return self.pages


class FakeEmbedder:
    """Return a deterministic vector for service tests."""

    def embed(self, text: str) -> tuple[float, ...]:
        return (1.0, 0.0)


def make_service(
    pages: tuple[DocumentPage, ...],
) -> tuple[IngestionService, FakeVectorStore]:
    target = FakeVectorStore()
    return (
        IngestionService(
            loader=FakeLoader(pages),
            embedder=FakeEmbedder(),
            vector_store=target,
            chunk_size=4,
            overlap=1,
        ),
        target,
    )


def test_ingest_chunks_embeds_and_persists_metadata(tmp_path: Path) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("one two three four five six", encoding="utf-8")
    service, store = make_service(
        (DocumentPage(text=document.read_text(encoding="utf-8"), page_number=1),)
    )

    result = service.ingest(document)

    assert result.chunk_count == 2
    assert [chunk.text for chunk in result.chunks] == [
        "one two three four",
        "four five six",
    ]
    assert result.chunks[0].metadata == {
        "source": str(document),
        "page": 1,
        "chunk_id": "notes.txt:p1:c0",
    }
    assert len(store.chunks) == 2
    assert all(len(chunk.embedding) == 2 for chunk in store.chunks)


def test_empty_document_fails_before_persisting(tmp_path: Path) -> None:
    document = tmp_path / "empty.txt"
    document.write_text(" \n", encoding="utf-8")
    service, store = make_service((DocumentPage(text="", page_number=1),))

    with pytest.raises(IngestionError, match="no extractable text"):
        service.ingest(document)

    assert store.chunks == []


def test_unsupported_file_type_is_explicit(tmp_path: Path) -> None:
    document = tmp_path / "archive.zip"
    document.write_bytes(b"not a document")

    with pytest.raises(UnsupportedFileTypeError, match="Unsupported document type"):
        LlamaIndexDocumentLoader().load(document)


def test_embed_failure_does_not_persist_partial_chunks(tmp_path: Path) -> None:
    class FailingEmbedder:
        def embed(self, text: str) -> tuple[float, ...]:
            if text.startswith("four"):
                raise RuntimeError("provider unavailable")
            return (1.0,)

    document = tmp_path / "notes.txt"
    document.write_text("one two three four five six", encoding="utf-8")
    store = FakeVectorStore()
    service = IngestionService(
        loader=FakeLoader(
            (DocumentPage(text=document.read_text(encoding="utf-8"), page_number=1),)
        ),
        embedder=FailingEmbedder(),
        vector_store=store,
        chunk_size=3,
        overlap=0,
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        service.ingest(document)

    assert store.chunks == []


def test_llama_index_chunker_uses_sentence_splitter(tmp_path: Path) -> None:
    chunker = LlamaIndexChunker(chunk_size=4, overlap=1)
    pages = (
        DocumentPage(
            text="one two three four five six seven eight nine ten",
            page_number=1,
        ),
    )

    chunks = chunker(pages, tmp_path / "notes.txt")

    assert len(chunks) >= 2
    assert all(chunk.source.endswith("notes.txt") for chunk in chunks)
    assert all(chunk.page_number == 1 for chunk in chunks)


def test_llama_index_embedder_delegates_to_model() -> None:
    class Model:
        def get_text_embedding(self, text: str) -> list[float]:
            assert text == "hello"
            return [0.1, 0.2]

    assert LlamaIndexEmbedder(Model()).embed("hello") == (0.1, 0.2)


def test_llama_index_loader_uses_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    document = tmp_path / "notes.txt"
    document.write_text("hello", encoding="utf-8")

    class Loaded:
        def __init__(self) -> None:
            self.metadata = {"page_label": "3"}

        def get_content(self) -> str:
            return "hello"

    class Reader:
        def __init__(self, **kwargs: object) -> None:
            assert kwargs["input_files"] == [document]

        def load_data(self) -> list[Loaded]:
            return [Loaded()]

    monkeypatch.setattr(
        "backend.ingestion.llama_index_adapter.SimpleDirectoryReader", Reader
    )
    assert LlamaIndexDocumentLoader().load(document)[0].page_number == 3
