"""Database monitoring helpers."""

from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session


def db_size(session: Session) -> dict[str, Any]:
    """Return database size and row counts for application tables."""
    size = session.execute(
        text("SELECT pg_size_pretty(pg_database_size(current_database()))")
    ).scalar_one()
    tables = (
        "users",
        "documents",
        "document_texts",
        "chats",
        "chat_documents",
        "messages",
    )
    counts: dict[str, int] = {}
    for table in tables:
        counts[table] = int(
            session.execute(text(f'SELECT count(*) FROM "{table}"')).scalar_one()
        )
    return {"database_size": size, "row_counts": counts}
