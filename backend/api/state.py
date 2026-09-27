"""Development-only in-memory state for API conversations."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from uuid import uuid4


@dataclass
class ChatState:
    """Mutable state belonging to one chat."""

    id: str
    title: str
    updated_at: datetime
    documents: list[str] = field(default_factory=list)
    message_count: int = 0


class InMemoryChatStore:
    """Thread-safe chat store suitable for local development and tests."""

    def __init__(self) -> None:
        self._chats: dict[str, ChatState] = {}
        self._lock = Lock()

    def create(self, title: str) -> ChatState:
        """Create and return a new chat."""
        chat = ChatState(uuid4().hex, title.strip(), datetime.now(UTC))
        with self._lock:
            self._chats[chat.id] = chat
        return chat

    def list(self) -> list[ChatState]:
        """Return chats newest first."""
        with self._lock:
            return sorted(
                self._chats.values(), key=lambda chat: chat.updated_at, reverse=True
            )

    def get(self, chat_id: str) -> ChatState | None:
        """Look up a chat by identifier."""
        with self._lock:
            return self._chats.get(chat_id)

    def touch(self, chat: ChatState) -> None:
        """Update a chat's modification timestamp."""
        with self._lock:
            chat.updated_at = datetime.now(UTC)
