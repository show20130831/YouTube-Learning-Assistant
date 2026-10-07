from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from yla.analysis.analyzer import Analyzer
from yla.config import AppSettings
from yla.db.models import Channel, LiveStatus, User, Video, VideoStatus
from yla.llm.client import ChatMessage, LLMResponse
from yla.pipeline.daily import run_daily_pipeline
from yla.pipeline.digest import collect_digest
from yla.pipeline.enrich import enrich_videos
from yla.sync import sync_config
from yla.youtube.data_api import MetadataError, VideoMetadata

pytestmark = pytest.mark.db

NOW = datetime.now(UTC)
LC = "UCC-lyoTfSrcJzA1ab3APAgw"
LONG = "A practical walkthrough of building production systems step by step with examples. " * 3


class FakeMetadata:
    def __init__(self, known: dict[str, tuple[int, LiveStatus]] | Exception) -> None:
        self.known = known
        self.calls: list[list[str]] = []

    def fetch(self, video_ids: Sequence[str]) -> dict[str, VideoMetadata]:
        self.calls.append(list(video_ids))
        if isinstance(self.known, Exception):
            raise self.known
        return {
            vid: VideoMetadata(video_id=vid, duration_seconds=d, live_status=live)
            for vid, (d, live) in self.known.items()
            if vid in video_ids
        }


def setup(session: Session, *, include_shorts: bool = False) -> User:
    settings = AppSettings.model_validate(
        {"user": {"include_shorts": include_shorts}, "channels": [{"handle": "LangChain", "channel_id": LC}]}
    )
    sync_config(session, settings)
    return session.scalars(select(User)).one()


def add(session: Session, vid: str, status: VideoStatus = VideoStatus.PENDING, days_ago: float = 0) -> Video:
    when = NOW - timedelta(days=days_ago)
    video = Video(
        youtube_video_id=vid,
        channel_id=session.scalars(select(Channel.id)).one(),
        title=f"Video {vid}",
        description=LONG,
        published_at=when,
        discovered_at=when,
        status=status,
    )
    session.add(video)
    session.flush()
    return video


def status(session: Session, vid: str) -> VideoStatus:
    return session.scalars(select(Video.status).where(Video.youtube_video_id == vid)).one()


def test_shorts_are_skipped_and_lengths_recorded(session: Session) -> None:
    user = setup(session)
    add(session, "short")
    long_video = add(session, "long")
    report = enrich_videos(
        session,
        user,
        FakeMetadata({"short": (54, LiveStatus.NONE), "long": (1800, LiveStatus.NONE)}),
        now=NOW,
    )

    assert report.shorts == ["short"]
    assert status(session, "short") is VideoStatus.SKIPPED_SHORT
    assert (long_video.status, long_video.duration_seconds, long_video.is_short) == (
        VideoStatus.PENDING,
        1800,
        False,
    )


def test_shorts_kept_when_user_wants_them(session: Session) -> None:
    user = setup(session, include_shorts=True)
    video = add(session, "short")
    enrich_videos(session, user, FakeMetadata({"short": (54, LiveStatus.NONE)}), now=NOW)
    assert (video.status, video.is_short) == (VideoStatus.PENDING, True)


def test_streams_wait_then_become_normal_videos(session: Session) -> None:
    user = setup(session)
    video = add(session, "stream")
    enrich_videos(session, user, FakeMetadata({"stream": (0, LiveStatus.UPCOMING)}), now=NOW)
    assert (video.status, video.live_status, video.duration_seconds) == (
        VideoStatus.WAITING,
        LiveStatus.UPCOMING,
        None,
    )

    report = enrich_videos(session, user, FakeMetadata({"stream": (5400, LiveStatus.NONE)}), now=NOW)
    assert report.released == ["stream"]
    assert (video.status, video.duration_seconds) == (VideoStatus.PENDING, 5400)


def test_streams_waiting_too_long_expire(session: Session) -> None:
    user = setup(session)
    add(session, "stuck", VideoStatus.WAITING, days_ago=4)
    report = enrich_videos(session, user, FakeMetadata({"stuck": (0, LiveStatus.LIVE)}), now=NOW)
    assert report.expired == ["stuck"]
    assert status(session, "stuck") is VideoStatus.FAILED


def test_missing_videos_become_unavailable(session: Session) -> None:
    user = setup(session)
    add(session, "gone")
    assert enrich_videos(session, user, FakeMetadata({}), now=NOW).missing == ["gone"]
    assert status(session, "gone") is VideoStatus.UNAVAILABLE


def test_only_unchecked_or_waiting_videos_are_queried(session: Session) -> None:
    user = setup(session)
    add(session, "new")
    add(session, "waiting", VideoStatus.WAITING)
    add(session, "done", VideoStatus.ANALYZED)
    add(session, "backfill", VideoStatus.SKIPPED_BACKFILL)
    known = add(session, "known")
    known.duration_seconds = 600
    session.flush()

    provider = FakeMetadata({})
    enrich_videos(session, user, provider, now=NOW)
    assert sorted(provider.calls[0]) == ["new", "waiting"]


def test_nothing_to_check_makes_no_request(session: Session) -> None:
    user = setup(session)
    provider = FakeMetadata({})
    assert enrich_videos(session, user, provider, now=NOW).checked == 0
    assert provider.calls == []


def test_api_failure_leaves_videos_untouched(session: Session) -> None:
    user = setup(session)
    video = add(session, "a")
    report = enrich_videos(session, user, FakeMetadata(MetadataError("HTTP 403: quotaExceeded")), now=NOW)
    assert report.error == "HTTP 403: quotaExceeded"
    assert (video.status, video.duration_seconds) == (VideoStatus.PENDING, None)


class OkLLM:
    def complete_json(
        self, messages: list[ChatMessage], *, schema: dict[str, Any], schema_name: str, model: str
    ) -> LLMResponse:
        data = {
            "one_line_summary": "s",
            "key_points": ["p"],
            "key_concepts": [],
            "keywords": ["k"],
            "matched_topics": [],
            "relevance_score": 10,
            "limitation": None,
        }
        return LLMResponse(data=data, model=model)


def test_daily_pipeline_skips_shorts_and_digest_counts_them(session: Session) -> None:
    settings = AppSettings.model_validate(
        {"channels": [{"handle": "LangChain", "channel_id": LC}], "llm": {"primary_model": "m"}}
    )
    sync_config(session, settings)
    user = session.scalars(select(User)).one()
    add(session, "short")
    add(session, "long")
    metadata = FakeMetadata({"short": (40, LiveStatus.NONE), "long": (900, LiveStatus.NONE)})

    report = run_daily_pipeline(
        session, user, settings, Analyzer(OkLLM(), ["m"], []), now=NOW, metadata=metadata
    )
    assert (report.analyzed, report.shorts_skipped) == (["long"], 1)

    data = collect_digest(session, user, now=NOW)
    assert (data.new_videos, data.shorts_skipped) == (1, 1)
