"""FastAPI application factory and route handlers."""

import logging
import os
import re
from pathlib import Path
from uuid import uuid4

from chromadb.errors import ChromaError
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.classification.service import FailureType
from backend.generation.types import GenerationError
from backend.ingestion.types import IngestionError
from backend.retrieval.types import RetrievalError
from backend.verification.types import VerificationError

from .dependencies import ApiServices, build_default_services, get_services
from .schemas import (
    ChatSummary,
    CreateChatRequest,
    DocumentUploadResponse,
    QuestionRequest,
    QuestionResponse,
    SourceReference,
    VerificationResponse,
)
from .state import ChatState, InMemoryChatStore

logger = logging.getLogger(__name__)


def _summary(chat: ChatState) -> ChatSummary:
    return ChatSummary(
        id=chat.id,
        title=chat.title,
        updated_at=chat.updated_at,
        documents=list(chat.documents),
        message_count=chat.message_count,
    )


def create_app(
    services: ApiServices | None = None,
    chat_store: InMemoryChatStore | None = None,
) -> FastAPI:
    """Build the API app with injectable pipeline services and chat state."""
    app = FastAPI(title="AOA API", version="0.1.0")
    app.state.chat_store = chat_store or InMemoryChatStore()
    app.state.services = services
    if services is None and os.getenv("GROQ_API_KEY"):
        app.state.services = build_default_services()
    app.state.upload_dir = Path(".api_uploads")

    origins = os.getenv(
        "CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[origin.strip() for origin in origins if origin.strip()],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        """Report API availability."""
        return {"status": "ok"}

    @app.post("/chats", response_model=ChatSummary, status_code=201)
    def create_chat(payload: CreateChatRequest) -> ChatSummary:
        """Create a new in-memory conversation."""
        return _summary(app.state.chat_store.create(payload.title))

    @app.get("/chats", response_model=list[ChatSummary])
    def list_chats() -> list[ChatSummary]:
        """List conversations newest first."""
        return [_summary(chat) for chat in app.state.chat_store.list()]

    @app.post(
        "/chats/{chat_id}/documents",
        response_model=DocumentUploadResponse,
    )
    async def upload_documents(
        chat_id: str,
        files: list[UploadFile] = File(...),  # noqa: B008
        pipeline: ApiServices = Depends(get_services),  # noqa: B008
    ) -> DocumentUploadResponse:
        """Ingest uploaded files into the selected conversation."""
        chat = app.state.chat_store.get(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")
        if not files:
            raise HTTPException(status_code=400, detail="At least one file is required")

        names: list[str] = []
        chunks = 0
        for upload in files:
            name = Path(upload.filename or "upload").name
            if not re.fullmatch(r"[A-Za-z0-9._ -]+", name):
                raise HTTPException(status_code=400, detail="Invalid filename")
            suffix = Path(name).suffix or ".txt"
            temp_path = app.state.upload_dir / f"{uuid4().hex}{suffix}"
            try:
                content = await upload.read()
                app.state.upload_dir.mkdir(parents=True, exist_ok=True)
                temp_path.write_bytes(content)
                result = pipeline.ingestion.ingest(
                    temp_path, metadata={"chat_id": chat_id}
                )
            except (
                IngestionError,
                ChromaError,
                OSError,
                RuntimeError,
                ValueError,
            ) as exc:
                logger.exception("Document ingestion failed for chat %s", chat_id)
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            finally:
                await upload.close()
                temp_path.unlink(missing_ok=True)
            names.append(name)
            chunks += result.chunk_count

        chat.documents.extend(names)
        app.state.chat_store.touch(chat)
        return DocumentUploadResponse(files=names, chunk_count=chunks)

    @app.post(
        "/chats/{chat_id}/questions",
        response_model=QuestionResponse,
    )
    def ask_question(
        chat_id: str,
        payload: QuestionRequest,
        pipeline: ApiServices = Depends(get_services),  # noqa: B008
    ) -> QuestionResponse:
        """Retrieve context and generate an answer for a conversation."""
        chat = app.state.chat_store.get(chat_id)
        if chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")
        try:
            retrieved = pipeline.retrieval.retrieve(
                payload.query, filters={"chat_id": chat_id}
            )
            answer = pipeline.generation.generate(
                payload.query, retrieved.retrieved_chunks
            )
            verification = pipeline.verification.verify(
                payload.query,
                answer.answer,
                retrieved.retrieved_chunks,
            )
            remediation_result = None
            if pipeline.remediation is not None and payload.web_search:
                remediation_result = pipeline.remediation.remediate(
                    payload.query,
                    answer.answer,
                    retrieved.retrieved_chunks,
                    (
                        retrieved.retrieval_scores[0]
                        if retrieved.retrieval_scores
                        else None
                    ),
                    verification,
                    pipeline.generation.generate_from_web,
                    pipeline.generation.regenerate,
                    pipeline.verification.verify,
                    FailureType.RETRIEVAL_GAP,
                )
        except (RetrievalError, GenerationError, VerificationError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        chat.message_count += 2
        app.state.chat_store.touch(chat)
        source_chunks = (
            remediation_result.source_chunks
            if remediation_result
            else retrieved.retrieved_chunks
        )
        source_scores = (
            tuple(0.0 for _ in source_chunks)
            if remediation_result
            and remediation_result.source_label == "not found in your documents"
            else retrieved.retrieval_scores
        )
        sources = [
            SourceReference(text=text, score=score)
            for text, score in zip(source_chunks, source_scores, strict=True)
        ]
        return QuestionResponse(
            query=payload.query,
            answer=remediation_result.answer if remediation_result else answer.answer,
            sources=sources,
            verification=VerificationResponse(
                passed=(
                    remediation_result.verification.passed
                    if remediation_result
                    else verification.passed
                ),
                verified=(
                    remediation_result.verification.verified
                    if remediation_result
                    else verification.verified
                ),
                score=(
                    remediation_result.verification.score
                    if remediation_result
                    else verification.score
                ),
                explanation=(
                    remediation_result.verification.explanation
                    if remediation_result
                    else verification.explanation
                ),
            ),
            failure_type=(
                remediation_result.failure_type.value
                if remediation_result
                and isinstance(remediation_result.failure_type, FailureType)
                else None
            ),
            remediation=remediation_result.remediation if remediation_result else None,
            source_label=(
                remediation_result.source_label
                if remediation_result
                else "document-grounded"
            ),
            remediation_error=(
                remediation_result.error if remediation_result else None
            ),
        )

    return app
