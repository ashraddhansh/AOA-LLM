"""Dependency injection contracts for API application services."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastapi import HTTPException, Request

from backend.generation.groq import GroqGenerator
from backend.generation.service import GenerationService
from backend.generation.types import GenerationResult
from backend.ingestion.llama_index_adapter import (
    LlamaIndexChromaStore,
    LlamaIndexEmbedder,
    LocalHashEmbedding,
)
from backend.ingestion.service import IngestionService
from backend.ingestion.types import IngestionResult
from backend.retrieval.service import RetrievalService
from backend.retrieval.types import RetrievalResult
from backend.verification.lynx import LynxVerifier
from backend.verification.types import VerificationResult


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


class VerificationPipeline(Protocol):
    """Minimal faithfulness verification interface required by the API."""

    def verify(
        self, query: str, answer: str, retrieved_chunks: Sequence[str]
    ) -> VerificationResult:
        """Verify an answer against retrieved context."""


@dataclass(frozen=True)
class ApiServices:
    """Pipeline services used by API endpoints."""

    ingestion: IngestionPipeline
    retrieval: RetrievalPipeline
    generation: GenerationPipeline
    verification: VerificationPipeline


def build_default_services() -> ApiServices:
    """Build the production pipeline using local embeddings, Groq, and Chroma."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required to configure the RAG pipeline.")
    if not os.getenv("PATRONUS_API_KEY"):
        raise RuntimeError(
            "PATRONUS_API_KEY is required to configure the verification pipeline."
        )

    embedder = LlamaIndexEmbedder(LocalHashEmbedding())
    store = LlamaIndexChromaStore.from_persist_dir(
        Path(os.getenv("CHROMA_PERSIST_DIR", ".api_data/chroma")),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "aoa_documents_local_v1"),
    )
    return ApiServices(
        ingestion=IngestionService(
            loader=None,
            embedder=embedder,
            vector_store=store,
        ),
        retrieval=RetrievalService(embedder=embedder, vector_store=store),
        generation=GenerationService(
            GroqGenerator(
                api_key=api_key,
                model=os.getenv("GROQ_MODEL"),
            )
        ),
        verification=LynxVerifier(),
    )


def get_services(request: Request) -> ApiServices:
    """Resolve pipeline services from application state for injection."""
    services = request.app.state.services
    if services is None:
        raise HTTPException(status_code=503, detail="Pipeline services unavailable")
    return services
