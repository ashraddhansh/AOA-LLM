"""SQLAlchemy engine and short-lived session helpers."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from backend.config import database_url


@lru_cache(maxsize=4)
def session_factory(url: str | None = None) -> sessionmaker[Session]:
    """Create a session factory for the configured or supplied database."""
    engine = create_engine(url or database_url(), pool_pre_ping=True)
    return sessionmaker(bind=engine, class_=Session, expire_on_commit=False)


@contextmanager
def get_session(url: str | None = None) -> Iterator[Session]:
    """Yield a transactional session and roll back on failure."""
    factory = session_factory(url)
    with factory() as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    """Create an engine without exposing its URL."""
    return create_engine(url or database_url(), pool_pre_ping=True)
