"""FastAPI application factory and route handlers."""

import logging
import os
import re
from pathlib import Path
from uuid import UUID, uuid4

from chromadb.errors import ChromaError
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import func, select

from backend.auth import google_profile, hash_password, issue_token, verify_password
from backend.db.models import Chat, Document, User
from backend.db.repositories import (
    append_message,
    create_user,
    list_recent_messages,
)
from backend.db.repositories import (
    create_chat as create_db_chat,
)
from backend.db.repositories import (
    list_chats as list_db_chats,
)
from backend.db.session import get_session
from backend.generation.types import GenerationError
from backend.ingestion.types import IngestionError
from backend.orchestration.question_graph import QuestionState
from backend.retrieval.types import RetrievalError
from backend.verification.types import VerificationError

from .dependencies import (
    ApiServices,
    build_default_services,
    get_current_user,
    get_services,
)
from .schemas import (
    AuthResponse,
    ChatSummary,
    CreateChatRequest,
    DocumentUploadResponse,
    GoogleLoginRequest,
    LoginRequest,
    MessageResponse,
    QuestionRequest,
    QuestionResponse,
    RegisterRequest,
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
    if services is None and (
        os.getenv("GENERATION_MODEL")
        or os.getenv("GENERATION_API_KEY")
        or os.getenv("GROQ_API_KEY")
    ):
        app.state.services = build_default_services()
    app.state.upload_dir = Path(".api_uploads")

    def db_enabled() -> bool:
        return bool(os.getenv("DATABASE_URL"))

    @app.post("/auth/register", response_model=AuthResponse, status_code=201)
    def register(payload: RegisterRequest) -> AuthResponse:
        """Create an email/password account."""
        if not db_enabled():
            raise HTTPException(status_code=503, detail="Database is not configured")
        with get_session() as session:
            existing = session.scalar(
                select(User).where(func.lower(User.email) == payload.email.lower())
            )
            if existing is not None:
                raise HTTPException(
                    status_code=409, detail="Email is already registered"
                )
            user = create_user(session, payload.email, hash_password(payload.password))
            return AuthResponse(
                access_token=issue_token(user), user_id=str(user.id), email=user.email
            )

    @app.post("/auth/login", response_model=AuthResponse)
    def login(payload: LoginRequest) -> AuthResponse:
        """Authenticate an email/password account."""
        if not db_enabled():
            raise HTTPException(status_code=503, detail="Database is not configured")
        with get_session() as session:
            user = session.scalar(
                select(User).where(func.lower(User.email) == payload.email.lower())
            )
            if user is None or not verify_password(
                payload.password, user.password_hash
            ):
                raise HTTPException(status_code=401, detail="Invalid email or password")
            return AuthResponse(
                access_token=issue_token(user), user_id=str(user.id), email=user.email
            )

    @app.post("/auth/google", response_model=AuthResponse)
    async def google_login(payload: GoogleLoginRequest) -> AuthResponse:
        """Sign in or create an account from a validated Google identity token."""
        if not db_enabled():
            raise HTTPException(status_code=503, detail="Database is not configured")
        subject, email = await google_profile(payload.id_token)
        with get_session() as session:
            user = session.scalar(select(User).where(User.google_subject == subject))
            if user is None:
                user = session.scalar(
                    select(User).where(func.lower(User.email) == email)
                )
                if user is None:
                    user = create_user(session, email)
                user.google_subject = subject
                session.flush()
            return AuthResponse(
                access_token=issue_token(user), user_id=str(user.id), email=user.email
            )

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
    def create_chat(
        payload: CreateChatRequest,
        current_user: User | None = Depends(get_current_user),  # noqa: B008
    ) -> ChatSummary:
        """Create a new in-memory conversation."""
        if db_enabled():
            assert current_user is not None
            with get_session() as session:
                chat = create_db_chat(session, current_user.id, payload.title)
                return ChatSummary(
                    id=str(chat.id),
                    title=chat.title,
                    updated_at=chat.updated_at,
                    documents=[],
                    message_count=0,
                )
        return _summary(app.state.chat_store.create(payload.title))

    @app.get("/chats", response_model=list[ChatSummary])
    def list_chats(
        current_user: User | None = Depends(get_current_user),  # noqa: B008
    ) -> list[ChatSummary]:
        """List conversations newest first."""
        if db_enabled():
            assert current_user is not None
            with get_session() as session:
                return [
                    ChatSummary(
                        id=str(chat.id),
                        title=chat.title,
                        updated_at=chat.updated_at,
                        documents=[document.filename for document in chat.documents],
                        message_count=len(chat.messages),
                    )
                    for chat in list_db_chats(session, current_user.id)
                ]
        return [_summary(chat) for chat in app.state.chat_store.list()]

    @app.get("/chats/{chat_id}/messages", response_model=list[MessageResponse])
    def get_messages(
        chat_id: str,
        current_user: User | None = Depends(get_current_user),  # noqa: B008
    ) -> list[MessageResponse]:
        """Load persisted messages for an owned chat."""
        if not db_enabled():
            return []
        assert current_user is not None
        with get_session() as session:
            try:
                rows = list_recent_messages(
                    session, UUID(chat_id), current_user.id, limit=100
                )
            except ValueError as exc:
                raise HTTPException(status_code=404, detail="Chat not found") from exc
            return [
                MessageResponse(
                    id=str(row.id),
                    role=row.role,
                    content=row.content,
                    created_at=row.created_at,
                    verification=(
                        VerificationResponse(
                            passed=row.lynx_verdict == "pass",
                            verified=bool(row.verification_verified),
                            score=row.verification_score,
                            explanation=row.verification_explanation,
                        )
                        if row.role == "assistant" and row.lynx_verdict is not None
                        else None
                    ),
                )
                for row in rows
            ]

    @app.post(
        "/chats/{chat_id}/documents",
        response_model=DocumentUploadResponse,
    )
    async def upload_documents(
        chat_id: str,
        files: list[UploadFile] = File(...),  # noqa: B008
        pipeline: ApiServices = Depends(get_services),  # noqa: B008
        current_user: User | None = Depends(get_current_user),  # noqa: B008
    ) -> DocumentUploadResponse:
        """Ingest uploaded files into the selected conversation."""
        memory_chat = app.state.chat_store.get(chat_id)
        if db_enabled():
            assert current_user is not None
            try:
                owned_chat_id = UUID(chat_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail="Chat not found") from exc
            with get_session() as session:
                if (
                    session.scalar(
                        select(Chat.id).where(
                            Chat.id == owned_chat_id, Chat.user_id == current_user.id
                        )
                    )
                    is None
                ):
                    raise HTTPException(status_code=404, detail="Chat not found")
        elif memory_chat is None:
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

        if db_enabled():
            assert current_user is not None
            with get_session() as session:
                db_chat = session.scalar(
                    select(Chat).where(
                        Chat.id == UUID(chat_id), Chat.user_id == current_user.id
                    )
                )
                if db_chat is None:
                    raise HTTPException(status_code=404, detail="Chat not found")
                for name in names:
                    document = Document(
                        user_id=current_user.id,
                        filename=name,
                        file_type=Path(name).suffix.lstrip(".") or "txt",
                        status="ready",
                    )
                    db_chat.documents.append(document)
                session.flush()
        else:
            assert memory_chat is not None
            memory_chat.documents.extend(names)
            app.state.chat_store.touch(memory_chat)
        return DocumentUploadResponse(files=names, chunk_count=chunks)

    @app.post(
        "/chats/{chat_id}/questions",
        response_model=QuestionResponse,
    )
    def ask_question(
        chat_id: str,
        payload: QuestionRequest,
        pipeline: ApiServices = Depends(get_services),  # noqa: B008
        current_user: User | None = Depends(get_current_user),  # noqa: B008
    ) -> QuestionResponse:
        """Retrieve context and generate an answer for a conversation."""
        chat = app.state.chat_store.get(chat_id)
        if db_enabled():
            assert current_user is not None
            try:
                owned_chat_id = UUID(chat_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail="Chat not found") from exc
            with get_session() as session:
                if (
                    session.scalar(
                        select(Chat.id).where(
                            Chat.id == owned_chat_id, Chat.user_id == current_user.id
                        )
                    )
                    is None
                ):
                    raise HTTPException(status_code=404, detail="Chat not found")
        elif chat is None:
            raise HTTPException(status_code=404, detail="Chat not found")
        try:
            state = pipeline.question_graph.invoke(
                QuestionState(
                    query=payload.query,
                    chat_id=chat_id,
                    web_search=payload.web_search,
                )
            )
        except (RetrievalError, GenerationError, VerificationError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        if not db_enabled():
            assert chat is not None
            chat.message_count += 2
            app.state.chat_store.touch(chat)
        source_chunks = state["source_chunks"]
        source_scores = state["source_scores"]
        sources = [
            SourceReference(text=text, score=score)
            for text, score in zip(source_chunks, source_scores, strict=True)
        ]
        verification = state["verification"]
        failure_type = state.get("failure_type")
        response = QuestionResponse(
            query=payload.query,
            answer=state["answer"],
            sources=sources,
            verification=VerificationResponse(
                passed=verification.passed,
                verified=verification.verified,
                score=verification.score,
                explanation=verification.explanation,
            ),
            failure_type=failure_type.value if failure_type else None,
            remediation=state.get("remediation"),
            source_label=state["source_label"],
            remediation_error=state.get("remediation_error"),
        )
        if db_enabled():
            assert current_user is not None
            try:
                owned_chat_id = UUID(chat_id)
            except ValueError as exc:
                raise HTTPException(status_code=404, detail="Chat not found") from exc
            with get_session() as session:
                user_message = append_message(
                    session,
                    owned_chat_id,
                    current_user.id,
                    "user",
                    payload.query,
                )
                assistant_message = append_message(
                    session,
                    owned_chat_id,
                    current_user.id,
                    "assistant",
                    response.answer,
                    source_type=(
                        "web"
                        if response.remediation == "web_search"
                        else "abstain"
                        if response.remediation == "abstain"
                        else "document"
                    ),
                    lynx_verdict=("pass" if response.verification.passed else "fail"),
                    verification_verified=response.verification.verified,
                    verification_score=response.verification.score,
                    verification_explanation=response.verification.explanation,
                    failure_type=response.failure_type or "none",
                    retrieval_scores=[source.score for source in sources],
                    retrieved_chunk_ids=[],
                    web_sources=[],
                )
                if user_message is None or assistant_message is None:
                    raise HTTPException(status_code=404, detail="Chat not found")
        return response

    return app
