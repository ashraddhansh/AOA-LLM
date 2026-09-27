"""Dependency injection contracts for API application services."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastapi import HTTPException, Request

from backend.generation.types import GenerationResult
from backend.ingestion.types import IngestionResult
from backend.retrieval.types import RetrievalResult


class IngestionPipeline(Protocol):
    """Minimal ingestion interface required by the API."""

    def ingest(self, path: Path) -> IngestionResult:
        """Ingest a file."""


class RetrievalPipeline(Protocol):
    """Minimal retrieval interface required by the API."""

    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve context."""


class GenerationPipeline(Protocol):
    """Minimal generation interface required by the API."""

    def generate(self, query: str, retrieved_chunks: Sequence[str]) -> GenerationResult:
        """Generate an answer."""


@dataclass(frozen=True)
class ApiServices:
    """Pipeline services used by API endpoints."""

    ingestion: IngestionPipeline
    retrieval: RetrievalPipeline
    generation: GenerationPipeline


def get_services(request: Request) -> ApiServices:
    """Resolve pipeline services from application state for injection."""
    services = request.app.state.services
    if services is None:
        raise HTTPException(status_code=503, detail="Pipeline services unavailable")
    return services
