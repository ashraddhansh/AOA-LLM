"""Pydantic request and response schemas for the HTTP API."""

from datetime import datetime

from pydantic import BaseModel, Field


class CreateChatRequest(BaseModel):
    """Payload used to create a conversation."""

    title: str = Field(default="New conversation", min_length=1, max_length=200)


class RegisterRequest(BaseModel):
    """Email/password registration payload."""

    email: str = Field(min_length=3, max_length=320)
    password: str = Field(min_length=8, max_length=200)


class LoginRequest(RegisterRequest):
    """Email/password login payload."""


class GoogleLoginRequest(BaseModel):
    """Google identity token payload."""

    id_token: str = Field(min_length=1)


class AuthResponse(BaseModel):
    """Authenticated user and bearer token."""

    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str


class MessageResponse(BaseModel):
    """Persisted chat message."""

    id: str
    role: str
    content: str
    created_at: datetime
    verification: "VerificationResponse | None" = None
    remediation: str | None = None
    source_label: str | None = None
    sources: list["SourceReference"] = Field(default_factory=list)


class ChatSummary(BaseModel):
    """Summary of a conversation and its uploaded documents."""

    id: str
    title: str
    updated_at: datetime
    documents: list[str]
    message_count: int


class QuestionRequest(BaseModel):
    """Payload containing a question and optional web-remediation request."""

    query: str = Field(min_length=1)
    web_search: bool = False


class SourceReference(BaseModel):
    """A source chunk used to answer a question."""

    text: str
    score: float


class VerificationResponse(BaseModel):
    """Faithfulness verification details for the generated answer."""

    passed: bool
    verified: bool
    score: float | None
    explanation: str | None


class QuestionResponse(BaseModel):
    """Answer and retrieval evidence returned to a client."""

    query: str
    answer: str
    sources: list[SourceReference]
    verification: VerificationResponse
    failure_type: str | None = None
    remediation: str | None = None
    source_label: str = "document-grounded"
    remediation_error: str | None = None


class DocumentUploadResponse(BaseModel):
    """Result of ingesting one or more uploaded documents."""

    files: list[str]
    chunk_count: int
