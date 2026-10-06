from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import respx
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from yla import cli
from yla.analysis.analyzer import Analyzer
from yla.config import AppSettings
from yla.db.models import Analysis, Channel, JobRun, JobStatus, Subscription, User, Video, VideoStatus
from yla.llm.client import OPENROUTER_URL, ChatMessage, LLMError, LLMRateLimited, LLMResponse
from yla.pipeline.daily import run_daily_pipeline
from yla.sync import sync_config
from yla.youtube.rss import Feed, FeedEntry
from yla.youtube.transcripts import DatabaseTranscriptSink, TranscriptResult, TranscriptStatus

pytestmark = pytest.mark.db

# Analyses get a DB-side created_at, so "today" must be the real today.
NOW = datetime.now(UTC)
LC = "UCC-lyoTfSrcJzA1ab3APAgw"
LONG = "A practical walkthrough of building production systems step by step with examples. " * 3
REPLY: dict[str, Any] = {
    "one_line_summary": "摘要",
    "key_points": ["重點"],
    "key_concepts": [],
    "keywords": ["Testing"],
    "matched_topics": [],
    "relevance_score": 10,
    "limitation": None,
}


class ScriptedLLM:
    """Replies per video title (found in the prompt); default is a valid reply."""

    def __init__(self, failures: dict[str, Exception] | None = None) -> None:
        self.failures = failures or {}
        self.titles: list[str] = []

    def complete_json(
        self, messages: list[ChatMessage], *, schema: dict[str, Any], schema_name: str, model: str
    ) -> LLMResponse:
        title = messages[1]["content"].splitlines()[0].removeprefix("影片標題：")
        self.titles.append(title)
        if title in self.failures:
            raise self.failures[title]
        return LLMResponse(data=REPLY, model=model)


def make_settings(limit: int = 5) -> AppSettings:
    return AppSettings.model_validate(
        {
            "user": {"daily_video_limit": limit},
            "channels": [{"handle": "LangChain", "channel_id": LC}],
            "topics": [{"name": "RAG"}, {"name": "AI Agent", "aliases": ["Agents"]}],
            "llm": {"primary_model": "m1", "fallback_models": ["m2"]},
        }
    )


def setup(session: Session, settings: AppSettings) -> User:
    sync_config(session, settings)
    return session.scalars(select(User)).one()


def add_video(
    session: Session,
    vid: str,
    *,
    title: str | None = None,
    description: str = LONG,
    days_ago: float = 0,
    status: VideoStatus = VideoStatus.PENDING,
    captions: bool = False,
) -> Video:
    when = NOW - timedelta(days=days_ago)
    video = Video(
        youtube_video_id=vid,
        channel_id=session.scalars(select(Channel.id)).one(),
        title=title or f"Video {vid}",
        description=description,
        published_at=when,
        discovered_at=when,
        status=status,
    )
    session.add(video)
    session.flush()
    if captions:
        result = TranscriptResult(status=TranscriptStatus.MANUAL, language="en", text="spoken words")
        DatabaseTranscriptSink(session).save(video.id, result)
    return video


def run(session: Session, settings: AppSettings, llm: ScriptedLLM, **kwargs: Any):  # type: ignore[no-untyped-def]
    user = session.scalars(select(User)).one()
    analyzer = Analyzer(llm, settings.llm.models, settings.topics)
    return run_daily_pipeline(session, user, settings, analyzer, now=kwargs.pop("now", NOW), **kwargs)


def status_of(session: Session, vid: str) -> VideoStatus:
    return session.scalars(select(Video.status).where(Video.youtube_video_id == vid)).one()


def test_grades_analyses_and_records_job(session: Session) -> None:
    settings = make_settings()
    setup(session, settings)
    add_video(session, "cap", captions=True, description="")
    add_video(session, "desc")
    add_video(session, "empty", description="#ai https://x.y")

    report = run(session, settings, ScriptedLLM())

    assert sorted(report.analyzed) == ["cap", "desc"]
    assert report.unavailable == ["empty"]
    assert report.llm_calls == 2
    assert status_of(session, "empty") is VideoStatus.UNAVAILABLE
    stars = dict(
        session.execute(select(Video.youtube_video_id, Analysis.confidence_stars).join(Analysis)).all()
    )
    assert stars == {"cap": "⭐⭐⭐", "desc": "⭐"}

    job = session.scalars(select(JobRun)).one()
    assert job.status is JobStatus.SUCCEEDED
    assert job.stats["analyzed"] == 2 and job.stats["run_id"] == report.run_id


def test_limit_prefers_topic_matches_and_defers_the_rest(session: Session) -> None:
    settings = make_settings(limit=2)
    setup(session, settings)
    add_video(session, "new_plain", days_ago=0)
    add_video(session, "older_rag", title="RAG pipelines explained", days_ago=1)
    add_video(session, "agents", title="Agents in production", days_ago=2)
    add_video(session, "old_plain", days_ago=2.5)

    report = run(session, settings, ScriptedLLM())

    assert report.analyzed == ["older_rag", "agents"]
    assert report.deferred == ["new_plain", "old_plain"]
    deferred = session.scalars(select(Video).where(Video.status == VideoStatus.DEFERRED)).all()
    assert all(v.deferred_since == NOW for v in deferred)


def test_rerun_same_day_does_not_exceed_limit_and_next_day_continues(session: Session) -> None:
    settings = make_settings(limit=1)
    setup(session, settings)
    add_video(session, "a")
    add_video(session, "b", days_ago=0.1)
    run(session, settings, ScriptedLLM())

    llm = ScriptedLLM()
    again = run(session, settings, llm)
    assert (again.analyzed, again.deferred, llm.titles) == ([], ["b"], [])

    tomorrow = run(session, settings, ScriptedLLM(), now=NOW + timedelta(days=1))
    assert tomorrow.analyzed == ["b"]


def test_videos_waiting_too_long_expire(session: Session) -> None:
    settings = make_settings()
    setup(session, settings)
    add_video(session, "stale", days_ago=4, status=VideoStatus.DEFERRED)

    report = run(session, settings, ScriptedLLM())
    assert report.expired == ["stale"]
    assert status_of(session, "stale") is VideoStatus.FAILED


def test_quota_exhausted_defers_everything_left(session: Session) -> None:
    settings = make_settings()
    setup(session, settings)
    add_video(session, "a")
    add_video(session, "b", days_ago=0.1)
    llm = ScriptedLLM({"Video a": LLMRateLimited("429")})

    report = run(session, settings, llm)

    assert report.quota_exhausted
    assert report.deferred == ["a", "b"]
    assert llm.titles == ["Video a", "Video a"]  # both models tried once, then the run stops
    video = session.scalars(select(Video).where(Video.youtube_video_id == "a")).one()
    assert video.attempts == 0  # not the video's fault


def test_other_failures_count_attempts_until_failed(session: Session) -> None:
    settings = make_settings()
    setup(session, settings)
    add_video(session, "bad")
    llm = ScriptedLLM({"Video bad": LLMError("HTTP 400")})

    for expected in (VideoStatus.DEFERRED, VideoStatus.DEFERRED, VideoStatus.FAILED):
        report = run(session, settings, llm)
        assert status_of(session, "bad") is expected
    assert report.failed == ["bad"]
    assert "HTTP 400" in (session.scalars(select(Video.last_error)).one() or "")


def test_disabled_channel_videos_are_ignored(session: Session) -> None:
    settings = make_settings()
    setup(session, settings)
    add_video(session, "a")
    sync_config(
        session,
        AppSettings.model_validate(
            {"channels": [{"handle": "LangChain", "channel_id": LC, "enabled": False}]}
        ),
    )
    assert run(session, settings, ScriptedLLM()).analyzed == []


class OneFeed:
    def fetch(self, channel_id: str) -> Feed:
        entry = FeedEntry(video_id="fresh", channel_id=LC, title="Fresh", description=LONG, published_at=NOW)
        return Feed(channel_title="LangChain", entries=[entry])


def test_discovery_runs_first(session: Session) -> None:
    settings = make_settings()
    setup(session, settings)
    # The feed was read before, so a new entry is a new video rather than backfill.
    session.scalars(select(Subscription)).one().last_checked_at = NOW - timedelta(days=1)
    report = run(session, settings, ScriptedLLM(), feed_fetcher=OneFeed())
    assert (report.discovered, report.analyzed) == (1, ["fresh"])


def test_unexpected_error_marks_job_failed(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_settings()
    setup(session, settings)

    def explode(*_: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr("yla.pipeline.daily._candidates", explode)

    with pytest.raises(RuntimeError):
        run(session, settings, ScriptedLLM())
    job = session.scalars(select(JobRun)).one()
    assert job.status is JobStatus.FAILED
    assert job.error == "RuntimeError: boom"


@pytest.fixture
def cli_db(engine, test_database_url, tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        f"channels: [{{handle: LangChain, channel_id: {LC}}}]\nllm: {{primary_model: 'm:free'}}\n",
        encoding="utf-8",
    )
    with Session(engine) as db:
        setup(db, make_settings())
        add_video(db, "v1")
        db.commit()
    yield engine, settings_file
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE users, channels, job_runs RESTART IDENTITY CASCADE"))


@respx.mock
def test_cli_run(cli_db: tuple[Engine, Path]) -> None:
    engine, settings_file = cli_db
    respx.post(OPENROUTER_URL).respond(
        200, json={"model": "m:free", "choices": [{"message": {"content": json.dumps(REPLY)}}]}
    )
    result = CliRunner().invoke(cli.app, ["run", "--skip-discover", "--path", str(settings_file)])

    assert result.exit_code == 0, result.output
    assert "analysed:    1  (1 LLM requests)" in result.output
    with Session(engine) as db:
        assert db.scalars(select(JobRun.status)).one() is JobStatus.SUCCEEDED


@respx.mock
def test_cli_run_reports_quota_exhausted(cli_db: tuple[Engine, Path]) -> None:
    respx.post(OPENROUTER_URL).respond(429, json={"error": {"code": 429, "message": "quota"}})
    result = CliRunner().invoke(cli.app, ["run", "--skip-discover", "--path", str(cli_db[1])])
    assert result.exit_code == 0, result.output
    assert "free quota appears used up" in result.output
