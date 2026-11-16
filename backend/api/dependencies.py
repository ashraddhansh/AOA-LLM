"""Dependency injection contracts for API application services."""

# FastAPI dependencies intentionally use call expressions in parameter defaults.
# ruff: noqa: B008

import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.generation.openai_compatible import OpenAICompatibleGenerator
from backend.generation.service import GenerationService
from backend.generation.types import GenerationPipeline
from backend.llm.openai_compatible import ChatCompletionClient
from backend.orchestration.question_graph import (
    QuestionGraphRunner,
    build_question_graph,
)
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
from backend.verification.types import VerificationResult
from backend.verification.openai_compatible import OpenAICompatibleVerifier

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
    question_graph: QuestionGraphRunner = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Compile the graph once and reuse it for each question request."""
        threshold = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.5"))
        object.__setattr__(
            self,
            "question_graph",
            build_question_graph(
                self.retrieval,
                self.generation,
                self.verification,
                self.remediation,
                threshold,
            ),
        )


bearer_scheme = HTTPBearer(auto_error=False)


def build_default_services() -> ApiServices:
    """Build Chroma, configurable chat-model, and optional search services."""

    generation_model = os.getenv("GENERATION_MODEL") or os.getenv(
        "GROQ_MODEL", "openai/gpt-oss-20b"
    )
    generation_base_url = os.getenv(
        "GENERATION_API_BASE_URL",
        os.getenv("GROQ_API_BASE_URL", "https://api.groq.com/openai/v1"),
    )
    generation_api_key = os.getenv("GENERATION_API_KEY") or os.getenv("GROQ_API_KEY")
    verification_model = os.getenv("VERIFICATION_MODEL") or generation_model
    verification_base_url = (
        os.getenv("VERIFICATION_API_BASE_URL") or generation_base_url
    )
    verification_api_key = (
        os.getenv("VERIFICATION_API_KEY") or generation_api_key
    )
    web_model = os.getenv("WEB_GENERATION_MODEL") or os.getenv(
        "GROQ_WEB_MODEL", generation_model
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
    search = (
        SerpApiSearch()
        if os.getenv("SERPAPI_API_KEY") or os.getenv("WEB_SEARCH_API_KEY")
        else None
    )
    return ApiServices(
        ingestion=IngestionService(
            loader=None,
            embedder=embedder,
            vector_store=store,
        ),
        retrieval=RetrievalService(embedder=embedder, vector_store=store),
        generation=GenerationService(
            OpenAICompatibleGenerator(
                ChatCompletionClient(
                    model=generation_model,
                    api_base_url=generation_base_url,
                    api_key=generation_api_key,
                )
            ),
            web_generator=OpenAICompatibleGenerator(
                ChatCompletionClient(
                    model=web_model,
                    api_base_url=generation_base_url,
                    api_key=generation_api_key,
                ),
            ),
        ),
        verification=OpenAICompatibleVerifier(
            ChatCompletionClient(
                model=verification_model,
                api_base_url=verification_base_url,
                api_key=verification_api_key,
            )
        ),
        remediation=RemediationService(search) if search else None,
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
