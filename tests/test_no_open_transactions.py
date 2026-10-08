"""Regression: no database transaction may stay open while waiting on the network.

Neon terminates connections that sit idle inside a transaction (2026-10-08: the cloud pipeline
failed with IdleInTransactionSessionTimeout while RSS retries took minutes). Every fake below
records whether the session was inside a transaction when the network call happened.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from yla.analysis.analyzer import Analyzer
from yla.config import AppSettings
from yla.db.models import Channel, LiveStatus, User, Video, VideoStatus
from yla.llm.client import ChatMessage, LLMResponse
from yla.pipeline.daily import run_daily_pipeline
from yla.pipeline.discovery import discover_videos
from yla.pipeline.enrich import enrich_videos
from yla.sync import sync_config
from yla.worker.local_worker import fetch_transcripts
from yla.youtube.data_api import VideoMetadata
from yla.youtube.rss import Feed, FeedEntry
from yla.youtube.transcripts import DatabaseTranscriptSink, TranscriptResult, TranscriptStatus

pytestmark = pytest.mark.db

NOW = datetime.now(UTC)
LC = "UCC-lyoTfSrcJzA1ab3APAgw"
LONG = "A practical walkthrough of building production systems step by step with examples. " * 3


class Probe:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.in_transaction: list[bool] = []

    def record(self) -> None:
        self.in_transaction.append(self.session.in_transaction())


def setup(session: Session, *, pending: int = 2) -> User:
    settings = AppSettings.model_validate(
        {"channels": [{"handle": "LangChain", "channel_id": LC}], "llm": {"primary_model": "m"}}
    )
    sync_config(session, settings)
    channel_id = session.scalars(select(Channel.id)).one()
    for i in range(pending):
        session.add(
            Video(
                youtube_video_id=f"v{i}",
                channel_id=channel_id,
                title=f"Video {i}",
                description=LONG,
                published_at=NOW,
                discovered_at=NOW,
                status=VideoStatus.PENDING,
            )
        )
    session.commit()
    return session.scalars(select(User)).one()


def test_discovery(session: Session) -> None:
    user = setup(session, pending=0)
    probe = Probe(session)

    class Fetcher:
        def fetch(self, channel_id: str) -> Feed:
            probe.record()
            entry = FeedEntry(video_id="new", channel_id=LC, title="t", description="d", published_at=NOW)
            return Feed(channel_title="LangChain", entries=[entry])

    discover_videos(session, user.id, Fetcher(), now=NOW)
    assert probe.in_transaction == [False]


def test_caption_fetching(session: Session) -> None:
    user = setup(session)
    probe = Probe(session)

    class Provider:
        def fetch(self, video_id: str) -> TranscriptResult:
            probe.record()
            return TranscriptResult(status=TranscriptStatus.GENERATED, language="en", text="words")

    fetch_transcripts(
        session, user.id, Provider(), DatabaseTranscriptSink(session), now=NOW, limit=5, sleep=lambda _: None
    )
    assert probe.in_transaction == [False, False]


def test_metadata_lookup(session: Session) -> None:
    user = setup(session)
    probe = Probe(session)

    class Metadata:
        def fetch(self, video_ids: Sequence[str]) -> dict[str, VideoMetadata]:
            probe.record()
            return {
                v: VideoMetadata(video_id=v, duration_seconds=900, live_status=LiveStatus.NONE)
                for v in video_ids
            }

    enrich_videos(session, user, Metadata(), now=NOW)
    assert probe.in_transaction == [False]


def test_llm_analysis(session: Session) -> None:
    user = setup(session)
    probe = Probe(session)

    class LLM:
        def complete_json(
            self, messages: list[ChatMessage], *, schema: dict[str, Any], schema_name: str, model: str
        ) -> LLMResponse:
            probe.record()
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

    settings = AppSettings.model_validate(
        {"channels": [{"handle": "LangChain", "channel_id": LC}], "llm": {"primary_model": "m"}}
    )
    report = run_daily_pipeline(session, user, settings, Analyzer(LLM(), ["m"], []), now=NOW)
    assert len(report.analyzed) == 2
    assert probe.in_transaction == [False, False]
