"""Shared fixtures.

Database tests run against a real Postgres (docker compose locally, a service container in CI)
in a separate ``yla_test`` database. Each test runs inside a transaction that is rolled back.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from yla.db.session import normalize_url

ROOT = Path(__file__).parents[1]
DEFAULT_TEST_DATABASE_URL = "postgresql+psycopg://yla:yla@localhost:5432/yla_test"


def _alembic_config(url: str) -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    cfg.set_main_option("sqlalchemy.url", url.replace("%", "%%"))
    return cfg


def _ensure_database(url: str) -> None:
    parsed = make_url(url)
    admin = create_engine(parsed.set(database="postgres"), isolation_level="AUTOCOMMIT")
    try:
        with admin.connect() as conn:
            exists = conn.scalar(text("SELECT 1 FROM pg_database WHERE datname = :n"), {"n": parsed.database})
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{parsed.database}"'))
    finally:
        admin.dispose()


@pytest.fixture(scope="session")
def test_database_url() -> str:
    return normalize_url(os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DATABASE_URL))


@pytest.fixture(scope="session")
def engine(test_database_url: str) -> Iterator[Engine]:
    try:
        _ensure_database(test_database_url)
    except OperationalError as exc:
        if os.environ.get("CI"):
            raise
        pytest.skip(f"Postgres not reachable ({exc.__class__.__name__}); run `docker compose up -d`")

    eng = create_engine(test_database_url)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))

    # Round-trip the migrations so a broken downgrade is caught too.
    cfg = _alembic_config(test_database_url)
    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
    command.upgrade(cfg, "head")

    yield eng
    eng.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with engine.connect() as conn:
        outer = conn.begin()
        db = Session(bind=conn, join_transaction_mode="create_savepoint")
        try:
            yield db
        finally:
            db.close()
            outer.rollback()
