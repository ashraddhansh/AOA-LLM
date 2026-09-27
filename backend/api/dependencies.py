"""Dependency injection contracts for API application services."""

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from fastapi import HTTPException, Request

from backend.generation.gemini import GeminiGenerator
from backend.generation.service import GenerationService
from backend.generation.types import GenerationResult
from backend.ingestion.llama_index_adapter import (
    LlamaIndexChromaStore,
    LlamaIndexEmbedder,
)
from backend.ingestion.service import IngestionService
from backend.ingestion.types import IngestionResult
from backend.retrieval.service import RetrievalService
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


def build_default_services() -> ApiServices:
    """Build the production pipeline using Gemini embeddings and Chroma."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is required to configure the RAG pipeline.")

    from llama_index.embeddings.gemini import GeminiEmbedding

    embedder = LlamaIndexEmbedder(
        GeminiEmbedding(
            model_name=os.getenv(
                "GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"
            ),
            api_key=api_key,
        )
    )
    store = LlamaIndexChromaStore.from_persist_dir(
        Path(os.getenv("CHROMA_PERSIST_DIR", ".api_data/chroma"))
    )
    return ApiServices(
        ingestion=IngestionService(
            loader=None,
            embedder=embedder,
            vector_store=store,
        ),
        retrieval=RetrievalService(embedder=embedder, vector_store=store),
        generation=GenerationService(GeminiGenerator(api_key=api_key)),
    )


def get_services(request: Request) -> ApiServices:
    """Resolve pipeline services from application state for injection."""
    services = request.app.state.services
    if services is None:
        raise HTTPException(status_code=503, detail="Pipeline services unavailable")
    return services
