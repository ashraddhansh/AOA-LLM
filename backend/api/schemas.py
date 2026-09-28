"""Pydantic request and response schemas for the HTTP API."""

from datetime import datetime

from pydantic import BaseModel, Field


class CreateChatRequest(BaseModel):
    """Payload used to create a conversation."""

    title: str = Field(default="New conversation", min_length=1, max_length=200)


class ChatSummary(BaseModel):
    """Summary of a conversation and its uploaded documents."""

    id: str
    title: str
    updated_at: datetime
    documents: list[str]
    message_count: int


class QuestionRequest(BaseModel):
    """Payload containing a document-grounded question."""

    query: str = Field(min_length=1)


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
    source_label: str = "uploaded documents"


class DocumentUploadResponse(BaseModel):
    """Result of ingesting one or more uploaded documents."""

    files: list[str]
    chunk_count: int
