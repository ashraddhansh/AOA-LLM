"""FastAPI application factory and route handlers."""

import os
import re
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.generation.types import GenerationError
from backend.ingestion.types import IngestionError
from backend.retrieval.types import RetrievalError

from .dependencies import ApiServices, build_default_services, get_services
from .schemas import (
    ChatSummary,
    CreateChatRequest,
    DocumentUploadResponse,
    QuestionRequest,
    QuestionResponse,
    SourceReference,
)
from .state import ChatState, InMemoryChatStore


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
    if services is None and os.getenv("GEMINI_API_KEY"):
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
                result = pipeline.ingestion.ingest(temp_path)
            except IngestionError as exc:
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
            retrieved = pipeline.retrieval.retrieve(payload.query)
            answer = pipeline.generation.generate(
                payload.query, retrieved.retrieved_chunks
            )
        except (RetrievalError, GenerationError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        chat.message_count += 2
        app.state.chat_store.touch(chat)
        sources = [
            SourceReference(text=text, score=score)
            for text, score in zip(
                retrieved.retrieved_chunks, retrieved.retrieval_scores, strict=True
            )
        ]
        return QuestionResponse(
            query=payload.query, answer=answer.answer, sources=sources
        )

    return app
