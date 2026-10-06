from datetime import time
from pathlib import Path

import pytest
from sqlalchemy import Engine, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from yla.cli import app
from yla.config import AppSettings, load_settings
from yla.db.models import Channel, Subscription, Topic, User
from yla.sync import sync_config

pytestmark = pytest.mark.db

EXAMPLE = Path(__file__).parents[1] / "config" / "settings.example.yaml"


def settings(channels: list[dict[str, object]], topics: list[dict[str, object]] | None = None) -> AppSettings:
    return AppSettings.model_validate({"channels": channels, "topics": topics or []})


def subscriptions(session: Session) -> dict[str, bool]:
    rows = session.execute(
        select(Channel.handle, Subscription.enabled).join(Subscription, Subscription.channel_id == Channel.id)
    )
    return {handle: enabled for handle, enabled in rows}


def topics(session: Session) -> dict[str, tuple[list[str], bool]]:
    return {t.name: (t.aliases, t.enabled) for t in session.scalars(select(Topic))}


def test_first_sync_creates_user_channels_and_topics(session: Session) -> None:
    report = sync_config(session, load_settings(EXAMPLE), line_user_id="U123")

    assert report.user_created
    assert len(report.channels_added) == 5
    assert len(report.topics_added) == 9
    user = session.scalars(select(User)).one()
    assert user.line_user_id == "U123"
    assert user.notify_time == time(12, 0)
    assert all(subscriptions(session).values())
    # Never-read feeds: the first RSS read will mark existing videos as backfill.
    assert all(s.last_checked_at is None for s in session.scalars(select(Subscription)))


def test_sync_is_idempotent(session: Session) -> None:
    cfg = load_settings(EXAMPLE)
    sync_config(session, cfg)
    report = sync_config(session, cfg)

    assert not report.user_created
    assert report.channels_added == report.channels_disabled == report.topics_added == []
    assert len(session.scalars(select(User)).all()) == 1


def test_removed_channel_is_disabled_and_readded_channel_is_enabled(session: Session) -> None:
    sync_config(session, settings([{"handle": "LangChain"}, {"handle": "statquest"}]))

    report = sync_config(session, settings([{"handle": "LangChain"}]))
    assert report.channels_disabled == ["statquest"]
    assert subscriptions(session) == {"LangChain": True, "statquest": False}

    report = sync_config(session, settings([{"handle": "LangChain"}, {"handle": "statquest"}]))
    assert report.channels_enabled == ["statquest"]
    assert subscriptions(session) == {"LangChain": True, "statquest": True}


def test_channel_disabled_in_yaml(session: Session) -> None:
    sync_config(session, settings([{"handle": "LangChain"}]))
    report = sync_config(session, settings([{"handle": "LangChain", "enabled": False}]))
    assert report.channels_disabled == ["LangChain"]
    assert subscriptions(session) == {"LangChain": False}


def test_handle_matching_is_case_insensitive(session: Session) -> None:
    sync_config(session, settings([{"handle": "LangChain"}]))
    report = sync_config(session, settings([{"handle": "@langchain"}]))
    assert report.channels_added == []
    assert len(session.scalars(select(Channel)).all()) == 1


def test_topics_are_updated_and_disabled(session: Session) -> None:
    sync_config(session, settings([{"handle": "x"}], [{"name": "RAG"}, {"name": "ML"}]))
    report = sync_config(session, settings([{"handle": "x"}], [{"name": "RAG", "aliases": ["Retrieval"]}]))

    assert report.topics_disabled == ["ML"]
    assert topics(session) == {"RAG": (["Retrieval"], True), "ML": ([], False)}


def test_user_preferences_are_updated(session: Session) -> None:
    sync_config(session, settings([{"handle": "x"}]))
    cfg = AppSettings.model_validate(
        {"channels": [{"handle": "x"}], "user": {"daily_video_limit": 3, "include_shorts": True}}
    )
    sync_config(session, cfg)
    user = session.scalars(select(User)).one()
    assert (user.daily_video_limit, user.include_shorts) == (3, True)


def test_db_rejects_duplicate_handle_in_different_case(session: Session) -> None:
    session.add(Channel(handle="LangChain"))
    session.flush()
    session.add(Channel(handle="langchain"))
    with pytest.raises(IntegrityError):
        session.flush()


def test_db_rejects_unknown_video_status(session: Session) -> None:
    session.add(Channel(handle="x"))
    session.flush()
    with pytest.raises(IntegrityError, match="ck_videos_videostatus"):
        session.execute(
            text(
                "INSERT INTO videos (youtube_video_id, channel_id, title, description, published_at,"
                " status, attempts) SELECT 'v1', id, 't', '', now(), 'bogus', 0 FROM channels"
            )
        )


def test_cli_sync_config_requires_database_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # no .env here
    monkeypatch.delenv("DATABASE_URL", raising=False)
    result = CliRunner().invoke(app, ["sync-config", "--path", str(EXAMPLE)])
    assert result.exit_code == 1
    assert "DATABASE_URL" in result.output


def test_cli_sync_config_writes_to_database(
    engine: Engine, test_database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    try:
        result = CliRunner().invoke(app, ["sync-config", "--path", str(EXAMPLE)])
        assert result.exit_code == 0, result.output
        assert "5 added" in result.output
        with Session(engine) as db:
            assert len(db.scalars(select(Subscription)).all()) == 5
    finally:
        with engine.begin() as conn:  # this test commits, so clean up explicitly
            conn.execute(text("TRUNCATE users, channels RESTART IDENTITY CASCADE"))
