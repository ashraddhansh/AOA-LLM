"""Dependency injection contracts for API application services."""

# FastAPI dependencies intentionally use call expressions in parameter defaults.
# ruff: noqa: B008

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.generation.groq import GroqGenerator
from backend.generation.service import GenerationService
from backend.generation.types import GenerationPipeline
from backend.ingestion.llama_index_adapter import (
    LlamaIndexChromaStore,
    LlamaIndexEmbedder,
    LocalHashEmbedding,
    migrate_legacy_chroma_collection,
)
from backend.ingestion.service import IngestionService
from backend.ingestion.types import IngestionResult
from backend.remediation.serpapi import SerpApiSearch
from backend.remediation.service import RemediationService

load_dotenv()
from backend.auth import user_from_token
from backend.db.models import User
from backend.db.session import get_session
from backend.retrieval.service import RetrievalService
from backend.retrieval.types import RetrievalResult
from backend.verification.lynx import LynxVerifier
from backend.verification.types import VerificationResult

logger = logging.getLogger(__name__)


class IngestionPipeline(Protocol):
    """Minimal ingestion interface required by the API."""

    def ingest(
        self, path: Path, metadata: dict[str, str | int] | None = None
    ) -> IngestionResult:
        """Ingest a file."""


class RetrievalPipeline(Protocol):
    """Minimal retrieval interface required by the API."""

    def retrieve(
        self, query: str, filters: dict[str, str | int] | None = None
    ) -> RetrievalResult:
        """Retrieve context."""


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
    remediation: RemediationService | None = None


bearer_scheme = HTTPBearer(auto_error=False)


def build_default_services() -> ApiServices:
    """Build the production pipeline using local embeddings, Groq, and Chroma."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is required to configure the RAG pipeline.")
    if not os.getenv("PATRONUS_API_KEY"):
        raise RuntimeError(
            "PATRONUS_API_KEY is required to configure the verification pipeline."
        )
    if not (os.getenv("SERPAPI_API_KEY") or os.getenv("WEB_SEARCH_API_KEY")):
        raise RuntimeError(
            "SERPAPI_API_KEY is required to configure web-search remediation."
        )

    embedder = LlamaIndexEmbedder(LocalHashEmbedding())
    store = LlamaIndexChromaStore.from_persist_dir(
        Path(os.getenv("CHROMA_PERSIST_DIR", ".api_data/chroma")),
        collection_name=os.getenv("CHROMA_COLLECTION_NAME", "aoa_documents_local_v2"),
    )
    migrated_chunks = migrate_legacy_chroma_collection(
        persist_dir=Path(os.getenv("CHROMA_PERSIST_DIR", ".api_data/chroma")),
        source_collection_name=os.getenv(
            "CHROMA_LEGACY_COLLECTION_NAME", "aoa_documents"
        ),
        target_collection_name=os.getenv(
            "CHROMA_COLLECTION_NAME", "aoa_documents_local_v2"
        ),
        embedder=embedder,
    )
    if migrated_chunks:
        logger.info(
            "Re-embedded %d chunks from the legacy Chroma collection",
            migrated_chunks,
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
            ),
            web_generator=GroqGenerator(
                api_key=api_key,
                model=os.getenv("GROQ_WEB_MODEL", "openai/gpt-oss-20b"),
                fallback_models=(
                    "qwen/qwen3-32b",
                    "meta-llama/llama-4-scout-17b-16e-instruct",
                ),
            ),
        ),
        verification=LynxVerifier(),
        remediation=RemediationService(
            SerpApiSearch(),
            retrieval_threshold=float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5")),
        ),
    )


def get_services(request: Request) -> ApiServices:
    """Resolve pipeline services from application state for injection."""
    services = request.app.state.services
    if services is None:
        raise HTTPException(status_code=503, detail="Pipeline services unavailable")
    return services


def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> User | None:
    """Resolve the bearer user when database authentication is configured."""
    if not os.getenv("DATABASE_URL"):
        return None
    if credentials is None:
        raise HTTPException(status_code=401, detail="Authentication required")
    with get_session() as session:
        user = user_from_token(session, credentials.credentials)
        session.expunge(user)
        return user
