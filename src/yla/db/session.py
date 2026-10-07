"""Engine and session helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import Engine, create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker


def normalize_url(url: str) -> str:
    """Use the psycopg 3 driver for plain ``postgresql://`` URLs (the form Neon hands out)."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url.removeprefix(prefix)
    return url


def connect_args(url: str) -> dict[str, Any]:
    """Driver options for the given URL.

    Behind a transaction-mode pooler (Neon's ``-pooler`` hosts, PgBouncer) a connection can
    change between statements, so psycopg's automatic server-side prepared statements are
    turned off there.
    """
    host = make_url(normalize_url(url)).host or ""
    return {"prepare_threshold": None} if "-pooler" in host else {}


def make_engine(url: str) -> Engine:
    return create_engine(normalize_url(url), pool_pre_ping=True, connect_args=connect_args(url))


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
