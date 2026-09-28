"""Environment-backed application configuration."""

import os
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from dotenv import load_dotenv

load_dotenv()


def normalize_database_url(url: str, *, external_ssl: bool = True) -> str:
    """Convert a PostgreSQL URL to the psycopg SQLAlchemy form."""
    if not url:
        raise ValueError("DATABASE_URL is required")
    parts = urlsplit(url)
    if parts.scheme not in {"postgres", "postgresql", "postgresql+psycopg"}:
        raise ValueError("DATABASE_URL must be a PostgreSQL URL")
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    if external_ssl and parts.hostname not in {"localhost", "127.0.0.1"}:
        query.setdefault("sslmode", "require")
    return urlunsplit(
        (
            "postgresql+psycopg",
            parts.netloc,
            parts.path,
            urlencode(query),
            parts.fragment,
        )
    )


def database_url(name: str = "DATABASE_URL", *, external_ssl: bool = True) -> str:
    """Read and normalize a database URL without logging its value."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} is required")
    return normalize_database_url(value, external_ssl=external_ssl)


@dataclass(frozen=True)
class Settings:
    """Database settings used by application and migration code."""

    database_url: str
    embed_dim: int = 256


def get_settings() -> Settings:
    """Load validated database settings from the environment."""
    raw_dim = os.getenv("EMBED_DIM", "256")
    try:
        embed_dim = int(raw_dim)
    except ValueError as exc:
        raise ValueError("EMBED_DIM must be an integer") from exc
    if embed_dim <= 0:
        raise ValueError("EMBED_DIM must be greater than zero")
    return Settings(database_url=database_url(), embed_dim=embed_dim)
