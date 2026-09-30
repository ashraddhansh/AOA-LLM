"""Ownership-enforcing repository operations."""

from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from sqlalchemy import case, select
from sqlalchemy.orm import Session

from .models import Chat, Document, DocumentText, Message, User
from .vector import delete_document_vectors


def create_user(session: Session, email: str, password_hash: str | None = None) -> User:
    """Create and flush a user."""
    user = User(email=email.lower(), password_hash=password_hash)
    session.add(user)
    session.flush()
    return user


def create_document(
    session: Session,
    user_id: UUID,
    filename: str,
    file_type: str,
    page_count: int | None = None,
) -> Document:
    """Create a document owned by the supplied user."""
    document = Document(
        user_id=user_id, filename=filename, file_type=file_type, page_count=page_count
    )
    session.add(document)
    session.flush()
    return document


def set_document_status(
    session: Session,
    document_id: UUID,
    user_id: UUID,
    status: str,
    error_message: str | None = None,
) -> Document | None:
    """Update status only when the document belongs to the user."""
    document = session.scalar(
        select(Document).where(Document.id == document_id, Document.user_id == user_id)
    )
    if document is None:
        return None
    document.status = status
    document.error_message = error_message
    session.flush()
    return document


def set_document_text(
    session: Session, document_id: UUID, user_id: UUID, extracted_text: str
) -> DocumentText | None:
    """Replace extracted text for an owned document."""
    owned = session.scalar(
        select(Document.id).where(
            Document.id == document_id, Document.user_id == user_id
        )
    )
    if owned is None:
        return None
    text = session.get(DocumentText, document_id)
    if text is None:
        text = DocumentText(document_id=document_id, extracted_text=extracted_text)
        session.add(text)
    else:
        text.extracted_text = extracted_text
    session.flush()
    return text


def list_documents(session: Session, user_id: UUID) -> list[Document]:
    """List only documents owned by a user."""
    return list(
        session.scalars(
            select(Document)
            .where(Document.user_id == user_id)
            .order_by(Document.created_at)
        )
    )


def delete_document(session: Session, document_id: UUID, user_id: UUID) -> bool:
    """Delete an owned document and its unreferenced vector rows."""
    document = session.scalar(
        select(Document).where(Document.id == document_id, Document.user_id == user_id)
    )
    if document is None:
        return False
    delete_document_vectors(session, document_id)
    session.delete(document)
    session.flush()
    return True


def create_chat(session: Session, user_id: UUID, title: str) -> Chat:
    """Create a chat owned by a user."""
    chat = Chat(user_id=user_id, title=title)
    session.add(chat)
    session.flush()
    return chat


def list_chats(session: Session, user_id: UUID) -> list[Chat]:
    """List only chats owned by a user, newest first."""
    return list(
        session.scalars(
            select(Chat).where(Chat.user_id == user_id).order_by(Chat.updated_at.desc())
        )
    )


def append_message(
    session: Session,
    chat_id: UUID,
    user_id: UUID,
    role: str,
    content: str,
    **metadata: Any,
) -> Message | None:
    """Append a message only to a chat owned by the user."""
    owned = session.scalar(
        select(Chat.id).where(Chat.id == chat_id, Chat.user_id == user_id)
    )
    if owned is None:
        return None
    message = Message(chat_id=chat_id, role=role, content=content, **metadata)
    chat = session.get(Chat, chat_id)
    if chat is not None:
        chat.updated_at = datetime.now(UTC)
        if role == "user" and chat.title == "New conversation":
            chat.title = content.strip()[:200] or chat.title
    session.add(message)
    session.flush()
    return message


def list_recent_messages(
    session: Session, chat_id: UUID, user_id: UUID, limit: int = 20
) -> list[Message]:
    """Return the most recent owned-chat messages in chronological order."""
    if limit <= 0:
        return []
    owned = session.scalar(
        select(Chat.id).where(Chat.id == chat_id, Chat.user_id == user_id)
    )
    if owned is None:
        return []
    rows = list(
        session.scalars(
            select(Message)
            .where(Message.chat_id == chat_id)
            # PostgreSQL's now() is transaction-scoped. The user and assistant
            # rows appended for one question therefore commonly share a
            # timestamp; UUID order is random and can put the answer first.
            .order_by(
                Message.created_at.desc(),
                case((Message.role == "user", 0), else_=1).desc(),
                Message.id.desc(),
            )
            .limit(limit)
        )
    )
    return list(reversed(rows))
