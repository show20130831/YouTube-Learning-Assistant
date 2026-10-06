"""Engine and session helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker


def normalize_url(url: str) -> str:
    """Use the psycopg 3 driver for plain ``postgresql://`` URLs (the form Neon hands out)."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


def make_engine(url: str) -> Engine:
    return create_engine(normalize_url(url), pool_pre_ping=True)


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Commit on success, roll back on error. Callers may also commit along the way."""
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    with factory() as session:
        try:
            yield session
            session.commit()
        except BaseException:
            session.rollback()
            raise
