from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from yla.cli import app
from yla.config import AppSettings
from yla.db.models import Channel, Subscription, Video, VideoStatus
from yla.pipeline.discovery import discover_videos
from yla.sync import sync_config
from yla.youtube.channels import resolve_missing_channel_ids
from yla.youtube.rss import RSS_URL, Feed, FeedEntry, FeedNotFoundError

pytestmark = pytest.mark.db

ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "config" / "settings.example.yaml"
FEED_XML = (ROOT / "tests" / "fixtures" / "rss" / "channel_feed.xml").read_bytes()
NOW = datetime(2026, 10, 6, 2, 30, tzinfo=UTC)
LC = "UCC-lyoTfSrcJzA1ab3APAgw"
SQ = "UCtYLUTtgS3k1Fg4y5tAhLbw"


def entry(video_id: str, channel_id: str = LC, days_ago: int = 0) -> FeedEntry:
    return FeedEntry(
        video_id=video_id,
        channel_id=channel_id,
        title=f"Video {video_id}",
        description="desc",
        published_at=NOW - timedelta(days=days_ago),
    )


class FakeFetcher:
    def __init__(self, feeds: dict[str, Feed | Exception]) -> None:
        self.feeds = feeds
        self.calls: list[str] = []

    def fetch(self, channel_id: str) -> Feed:
        self.calls.append(channel_id)
        result = self.feeds[channel_id]
        if isinstance(result, Exception):
            raise result
        return result


def setup_user(session: Session, channels: list[dict[str, object]]) -> int:
    sync_config(session, AppSettings.model_validate({"channels": channels}))
    return session.scalars(select(Subscription.user_id)).first() or 0


def statuses(session: Session) -> dict[str, VideoStatus]:
    return {v.youtube_video_id: v.status for v in session.scalars(select(Video))}


def test_first_read_backfills_and_later_reads_queue_new_videos(session: Session) -> None:
    user_id = setup_user(session, [{"handle": "LangChain", "channel_id": LC}])
    fetcher = FakeFetcher({LC: Feed(channel_title="LangChain", entries=[entry("old1", days_ago=3)])})

    report = discover_videos(session, user_id, fetcher, now=NOW)
    assert (report.backfilled, report.new_videos) == (1, [])
    assert statuses(session) == {"old1": VideoStatus.SKIPPED_BACKFILL}

    fetcher.feeds[LC] = Feed(channel_title="LangChain", entries=[entry("new1"), entry("old1", days_ago=3)])
    report = discover_videos(session, user_id, fetcher, now=NOW + timedelta(days=1))
    assert report.new_videos == ["new1"]
    assert statuses(session) == {"old1": VideoStatus.SKIPPED_BACKFILL, "new1": VideoStatus.PENDING}


def test_rerun_is_idempotent(session: Session) -> None:
    user_id = setup_user(session, [{"handle": "LangChain", "channel_id": LC}])
    fetcher = FakeFetcher({LC: Feed(channel_title=None, entries=[])})
    discover_videos(session, user_id, fetcher, now=NOW)
    fetcher.feeds[LC] = Feed(channel_title=None, entries=[entry("new1")])

    first = discover_videos(session, user_id, fetcher, now=NOW)
    second = discover_videos(session, user_id, fetcher, now=NOW)
    assert (first.new_videos, second.new_videos) == (["new1"], [])
    assert len(statuses(session)) == 1


def test_records_check_time_and_fills_channel_title(session: Session) -> None:
    user_id = setup_user(session, [{"handle": "LangChain", "channel_id": LC}])
    discover_videos(session, user_id, FakeFetcher({LC: Feed(channel_title="LangChain", entries=[])}), now=NOW)

    sub = session.scalars(select(Subscription)).one()
    assert sub.last_checked_at == NOW
    assert session.scalars(select(Channel.title)).one() == "LangChain"


def test_failed_channel_does_not_block_others(session: Session) -> None:
    user_id = setup_user(
        session, [{"handle": "LangChain", "channel_id": LC}, {"handle": "sq", "channel_id": SQ}]
    )
    fetcher = FakeFetcher(
        {LC: httpx.ConnectError("down"), SQ: Feed(channel_title=None, entries=[entry("s1", SQ)])}
    )

    report = discover_videos(session, user_id, fetcher, now=NOW)
    assert list(report.failed_channels) == ["LangChain"]
    assert report.channels_checked == 1
    checked = dict(
        session.execute(select(Channel.handle, Subscription.last_checked_at).join(Subscription)).all()
    )
    # The failed channel stays "never read", so its first successful read still backfills.
    assert checked == {"LangChain": None, "sq": NOW}


def test_not_found_feed_is_reported(session: Session) -> None:
    user_id = setup_user(session, [{"handle": "gone", "channel_id": LC}])
    report = discover_videos(session, user_id, FakeFetcher({LC: FeedNotFoundError("404")}), now=NOW)
    assert "FeedNotFoundError" in report.failed_channels["gone"]


def test_disabled_and_unresolved_channels_are_not_fetched(session: Session) -> None:
    user_id = setup_user(
        session,
        [
            {"handle": "LangChain", "channel_id": LC, "enabled": False},
            {"handle": "NoIdYet"},
        ],
    )
    fetcher = FakeFetcher({})
    report = discover_videos(session, user_id, fetcher, now=NOW)
    assert fetcher.calls == []
    assert report.skipped_without_id == ["NoIdYet"]


class FakeResolver:
    def __init__(self, known: dict[str, str]) -> None:
        self.known = known

    def resolve(self, handle: str) -> str | None:
        return self.known.get(handle)


def test_resolver_fills_missing_ids(session: Session) -> None:
    user_id = setup_user(
        session, [{"handle": "LangChain"}, {"handle": "Unknown"}, {"handle": "sq", "channel_id": SQ}]
    )

    report = resolve_missing_channel_ids(session, user_id, FakeResolver({"LangChain": LC}))
    assert report.resolved == {"LangChain": LC}
    assert report.unresolved == ["Unknown"]
    assert (
        session.scalars(select(Channel.youtube_channel_id).where(Channel.handle == "LangChain")).one() == LC
    )


def test_without_resolver_missing_ids_are_only_reported(session: Session) -> None:
    user_id = setup_user(session, [{"handle": "LangChain"}])
    report = resolve_missing_channel_ids(session, user_id, resolver=None)
    assert report.unresolved == ["LangChain"]
    assert session.scalars(select(Channel.youtube_channel_id)).one() is None


@respx.mock
def test_cli_discover_end_to_end(
    engine: Engine, test_database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    respx.get(RSS_URL).respond(200, content=FEED_XML)
    runner = CliRunner()
    try:
        assert runner.invoke(app, ["discover"]).exit_code == 1  # no user yet
        assert runner.invoke(app, ["sync-config", "--path", str(EXAMPLE)]).exit_code == 0

        result = runner.invoke(app, ["discover"])
        assert result.exit_code == 0, result.output
        assert "Checked 5 channel feeds" in result.output
        assert "backfilled:  3" in result.output  # the same 3 IDs in every mocked feed

        with Session(engine) as db:
            assert {v.status for v in db.scalars(select(Video))} == {VideoStatus.SKIPPED_BACKFILL}
    finally:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE users, channels RESTART IDENTITY CASCADE"))
