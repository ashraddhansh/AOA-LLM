"""Focused tests for the FastAPI HTTP boundary."""

from pathlib import Path

from fastapi.testclient import TestClient

from backend.api import create_app
from backend.api.dependencies import ApiServices
from backend.generation.types import GenerationResult
from backend.ingestion.types import IngestionResult
from backend.retrieval.types import RetrievalResult
from backend.verification.types import VerificationResult


class FakeIngestion:
    """Capture uploaded paths and return a deterministic result."""

    def __init__(self) -> None:
        self.paths: list[Path] = []
        self.metadata: list[dict[str, str | int] | None] = []

    def ingest(
        self, path: Path, metadata: dict[str, str | int] | None = None
    ) -> IngestionResult:
        self.paths.append(path)
        self.metadata.append(metadata)
        return IngestionResult(source=str(path), chunk_count=1, chunks=())


class FakeRetrieval:
    """Return one deterministic context chunk."""

    def __init__(self) -> None:
        self.filters: dict[str, str | int] | None = None

    def retrieve(
        self, query: str, filters: dict[str, str | int] | None = None
    ) -> RetrievalResult:
        self.filters = filters
        return RetrievalResult(query, ("source context",), (0.8,))


class FakeGeneration:
    """Return a deterministic answer."""

    def generate(self, query: str, chunks: tuple[str, ...]) -> GenerationResult:
        return GenerationResult(query, "grounded answer")


class FakeVerification:
    """Return a deterministic faithfulness result."""

    def verify(
        self, query: str, answer: str, chunks: tuple[str, ...]
    ) -> VerificationResult:
        return VerificationResult(True, True, 0.98, "Supported by context")


def make_client() -> tuple[TestClient, FakeIngestion, FakeRetrieval]:
    ingestion = FakeIngestion()
    retrieval = FakeRetrieval()
    services = ApiServices(ingestion, retrieval, FakeGeneration(), FakeVerification())
    return TestClient(create_app(services)), ingestion, retrieval


def test_chat_creation_and_listing() -> None:
    client, _, _ = make_client()

    created = client.post("/chats", json={"title": "Research"})

    assert created.status_code == 201
    chat = created.json()
    assert chat["title"] == "Research"
    listed = client.get("/chats")
    assert listed.status_code == 200
    assert listed.json()[0]["id"] == chat["id"]


def test_upload_and_question_use_injected_services() -> None:
    client, ingestion, retrieval = make_client()
    chat_id = client.post("/chats", json={}).json()["id"]

    upload = client.post(
        f"/chats/{chat_id}/documents",
        files={"files": ("notes.txt", b"hello document", "text/plain")},
    )
    assert upload.status_code == 200
    assert upload.json() == {"files": ["notes.txt"], "chunk_count": 1}
    assert len(ingestion.paths) == 1
    assert ingestion.metadata == [{"chat_id": chat_id}]
    assert not ingestion.paths[0].exists()

    answer = client.post(f"/chats/{chat_id}/questions", json={"query": "What is this?"})
    assert answer.status_code == 200
    assert answer.json()["answer"] == "grounded answer"
    assert answer.json()["sources"] == [{"text": "source context", "score": 0.8}]
    assert answer.json()["verification"]["passed"] is True
    assert retrieval.filters == {"chat_id": chat_id}


def test_unknown_chat_and_unconfigured_services_are_safe() -> None:
    client = TestClient(create_app())

    assert (
        client.post("/chats/missing/questions", json={"query": "x"}).status_code == 503
    )
    assert client.get("/health").json() == {"status": "ok"}


def test_cors_header_is_present() -> None:
    client, _, _ = make_client()

    response = client.get("/health", headers={"Origin": "http://localhost:5173"})

    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
