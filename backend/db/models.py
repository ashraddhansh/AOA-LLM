"""SQLAlchemy models for AOA application data."""

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    MetaData,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_name)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base with the project naming convention."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class DocumentStatus(StrEnum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class SourceType(StrEnum):
    DOCUMENT = "document"
    WEB = "web"
    ABSTAIN = "abstain"


class LynxVerdict(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


class FailureType(StrEnum):
    NONE = "none"
    RETRIEVAL_GAP = "retrieval_gap"
    BAD_GENERATION = "bad_generation"


class User(Base):
    """Application user."""

    __tablename__ = "users"

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    google_subject: Mapped[str | None] = mapped_column(String(255), unique=True)
    password_hash: Mapped[str | None] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    documents: Mapped[list["Document"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    chats: Mapped[list["Chat"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    __table_args__ = (Index("uq_users_email_lower", func.lower(email), unique=True),)


class Document(Base):
    """Uploaded document metadata."""

    __tablename__ = "documents"
    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(128), nullable=False)
    page_count: Mapped[int | None] = mapped_column()
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=DocumentStatus.PROCESSING.value
    )
    error_message: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    user: Mapped[User] = relationship(back_populates="documents")
    text: Mapped["DocumentText | None"] = relationship(
        back_populates="document", cascade="all, delete-orphan", uselist=False
    )
    chats: Mapped[list["Chat"]] = relationship(
        secondary="chat_documents", back_populates="documents"
    )

    __table_args__ = (
        Index("ix_documents_user_id_created_at", "user_id", "created_at"),
        CheckConstraint(
            "status IN ('processing', 'ready', 'failed')", name="document_status"
        ),
    )


class DocumentText(Base):
    """Extracted text kept separate from document list queries."""

    __tablename__ = "document_texts"

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    extracted_text: Mapped[str] = mapped_column(Text, nullable=False)
    document: Mapped[Document] = relationship(back_populates="text")


class Chat(Base):
    """User-owned chat."""

    __tablename__ = "chats"
    __table_args__ = (
        Index(
            "ix_chats_user_id_updated_at",
            "user_id",
            "updated_at",
            postgresql_using="btree",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    user_id: Mapped[UUID] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    user: Mapped[User] = relationship(back_populates="chats")
    documents: Mapped[list[Document]] = relationship(
        secondary="chat_documents", back_populates="chats"
    )
    messages: Mapped[list["Message"]] = relationship(
        back_populates="chat", cascade="all, delete-orphan"
    )


class ChatDocument(Base):
    """Association between a chat and an accessible document."""

    __tablename__ = "chat_documents"
    chat_id: Mapped[UUID] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )


class Message(Base):
    """Chat message and RAG evaluation metadata."""

    __tablename__ = "messages"
    __table_args__ = (
        Index("ix_messages_chat_id_created_at", "chat_id", "created_at"),
        CheckConstraint("role IN ('user', 'assistant')", name="message_role"),
        CheckConstraint(
            "source_type IS NULL OR source_type IN ('document', 'web', 'abstain')",
            name="message_source_type",
        ),
        CheckConstraint(
            "lynx_verdict IS NULL OR lynx_verdict IN ('pass', 'fail', 'error')",
            name="message_lynx_verdict",
        ),
        CheckConstraint(
            "failure_type IS NULL OR failure_type IN ('none', 'retrieval_gap', 'bad_generation')",
            name="message_failure_type",
        ),
    )

    id: Mapped[UUID] = mapped_column(
        PGUUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid()
    )
    chat_id: Mapped[UUID] = mapped_column(
        ForeignKey("chats.id", ondelete="CASCADE"), nullable=False
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    source_type: Mapped[str | None] = mapped_column(String(16))
    lynx_verdict: Mapped[str | None] = mapped_column(String(16))
    verification_verified: Mapped[bool | None] = mapped_column(Boolean)
    verification_score: Mapped[float | None] = mapped_column()
    verification_explanation: Mapped[str | None] = mapped_column(Text)
    failure_type: Mapped[str | None] = mapped_column(String(32))
    retrieval_scores: Mapped[Any | None] = mapped_column(JSONB)
    retrieved_chunk_ids: Mapped[Any | None] = mapped_column(JSONB)
    web_sources: Mapped[Any | None] = mapped_column(JSONB)
    regenerated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    chat: Mapped[Chat] = relationship(back_populates="messages")
