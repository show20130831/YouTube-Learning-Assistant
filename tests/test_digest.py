from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
import respx
from sqlalchemy import Engine, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from yla import cli
from yla.config import AppSettings
from yla.db.models import (
    Analysis,
    Channel,
    Digest,
    DigestStatus,
    JobRun,
    JobStatus,
    SourceType,
    User,
    UserVideoRelevance,
    Video,
    VideoStatus,
)
from yla.notify.line import PUSH_URL, NotifyError
from yla.pipeline.digest import collect_digest, send_digest
from yla.sync import sync_config

pytestmark = pytest.mark.db

NOW = datetime.now(UTC)  # analyses get a DB-side created_at, so use the real clock
LC = "UCC-lyoTfSrcJzA1ab3APAgw"
STARS = {3: "⭐⭐⭐", 2: "⭐⭐", 1: "⭐"}
SOURCES = {3: SourceType.MANUAL_CAPTION, 2: SourceType.AUTO_CAPTION, 1: SourceType.TITLE_DESCRIPTION}


def setup(session: Session) -> User:
    sync_config(
        session, AppSettings.model_validate({"channels": [{"handle": "LangChain", "channel_id": LC}]})
    )
    return session.scalars(select(User)).one()


def add_video(session: Session, vid: str, status: VideoStatus = VideoStatus.ANALYZED, **kw: object) -> Video:
    video = Video(
        youtube_video_id=vid,
        channel_id=session.scalars(select(Channel.id)).one(),
        title=f"Video {vid}",
        description="",
        published_at=NOW,
        discovered_at=kw.pop("discovered_at", NOW),
        status=status,
    )
    session.add(video)
    session.flush()
    return video


def analyse(
    session: Session, user: User, video: Video, *, level: int = 2, score: int = 50, at: datetime | None = None
) -> None:
    row = Analysis(
        video_id=video.id,
        model="m",
        prompt_version="p",
        source_type=SOURCES[level],
        confidence_level=level,
        confidence_stars=STARS[level],
        one_line_summary=f"summary {video.youtube_video_id}",
        key_points=["p"],
        key_concepts=[],
        keywords=["RAG"],
        raw_response={},
    )
    if at is not None:
        row.created_at = at
    session.add(row)
    values = {"relevance_score": score, "matched_topics": ["RAG"]}
    session.execute(
        insert(UserVideoRelevance)
        .values(user_id=user.id, video_id=video.id, **values)
        .on_conflict_do_update(index_elements=["user_id", "video_id"], set_=values)
    )
    session.flush()


def daily_job(session: Session, status: JobStatus = JobStatus.SUCCEEDED, **stats: object) -> None:
    session.add(
        JobRun(job_name="daily_pipeline", status=status, stats=stats, error="boom" if stats else None)
    )
    session.flush()


class FakeNotifier:
    def __init__(self, error: Exception | None = None, delivered: bool = True) -> None:
        self.error, self.delivered = error, delivered
        self.pushes: list[tuple[list[str], str]] = []

    def push(self, messages: list[str], *, retry_key: str) -> bool:
        self.pushes.append((messages, retry_key))
        if self.error:
            raise self.error
        return self.delivered


def worker_job(session: Session, status: JobStatus = JobStatus.SUCCEEDED, **stats: object) -> None:
    error = "ConnectionError: x" if status is JobStatus.FAILED else None
    session.add(JobRun(job_name="local_worker", status=status, stats=stats, error=error))
    session.flush()


def test_collects_todays_analyses_counts_and_trends(session: Session) -> None:
    user = setup(session)
    daily_job(session)
    worker_job(session, saved=2, attempted=2)
    analyse(session, user, add_video(session, "a"), level=3, score=80)
    analyse(session, user, add_video(session, "b"), level=1, score=40)
    add_video(session, "gone", status=VideoStatus.UNAVAILABLE)
    add_video(session, "later", status=VideoStatus.DEFERRED)
    add_video(session, "old", status=VideoStatus.SKIPPED_BACKFILL)

    data = collect_digest(session, user, now=NOW)

    assert {v.title: (v.confidence_level, v.relevance_score) for v in data.videos} == {
        "Video a": (3, 80),
        "Video b": (1, 40),
    }
    assert (data.channels_tracked, data.new_videos, data.deferred) == (1, 4, 1)
    assert data.unavailable == ["Video gone｜LangChain"]
    assert data.trends == [("RAG", 2)]
    assert data.notices == []


def test_reanalysed_video_shows_latest_and_is_marked_upgraded(session: Session) -> None:
    user = setup(session)
    daily_job(session)
    video = add_video(session, "a")
    analyse(session, user, video, level=1)
    analyse(session, user, video, level=2)

    [item] = collect_digest(session, user, now=NOW).videos
    assert (item.confidence_level, item.upgraded) == (2, True)


def test_window_starts_after_last_sent_digest(session: Session) -> None:
    user = setup(session)
    daily_job(session)
    analyse(session, user, add_video(session, "old"), at=NOW - timedelta(days=1, hours=2))
    analyse(session, user, add_video(session, "unsent"), at=NOW - timedelta(hours=20))
    yesterday = (NOW - timedelta(days=1)).date()
    session.add(
        Digest(
            user_id=user.id, digest_date=yesterday, status=DigestStatus.SENT, sent_at=NOW - timedelta(days=1)
        )
    )
    session.flush()

    titles = {v.title for v in collect_digest(session, user, now=NOW).videos}
    assert titles == {"Video unsent"}  # missed nothing after yesterday's push, repeated nothing before it


@pytest.mark.parametrize(
    ("job", "expected"),
    [
        (None, "⚠️ 今日分析流程沒有執行，請檢查排程。"),
        ((JobStatus.FAILED, {"x": 1}), "⚠️ 今日分析流程失敗：boom"),
        ((JobStatus.SUCCEEDED, {"quota_exhausted": True}), "⚠️ OpenRouter 免費額度已用完，0 部影片延到明天。"),
    ],
)
def test_pipeline_problems_become_notices(
    session: Session, job: tuple[JobStatus, dict[str, object]] | None, expected: str
) -> None:
    user = setup(session)
    if job:
        daily_job(session, job[0], **job[1])
    assert expected in collect_digest(session, user, now=NOW).notices


def test_successful_worker_is_an_info_line(session: Session) -> None:
    user = setup(session)
    daily_job(session)
    worker_job(session, saved=3, attempted=4, feed_errors={"LangChain": "x"})
    data = collect_digest(session, user, now=NOW)
    assert data.notices == []
    [line] = data.status
    assert line.startswith("💻 本機 worker：") and "完成（字幕 3/4，RSS 失敗 1 個頻道）" in line


@pytest.mark.parametrize(
    ("status", "stats", "expected"),
    [
        (None, {}, "⚠️ 本機 worker 今天沒有執行（電腦可能沒開機）"),
        (JobStatus.FAILED, {}, "⚠️ 本機 worker 執行失敗：ConnectionError: x"),
        (JobStatus.RUNNING, {}, "⚠️ 本機 worker 尚未執行完成"),
        (JobStatus.SUCCEEDED, {"blocked": True}, "⚠️ 本機 worker 的字幕請求被 YouTube 封鎖"),
    ],
)
def test_worker_problems_become_warnings(
    session: Session, status: JobStatus | None, stats: dict[str, object], expected: str
) -> None:
    user = setup(session)
    daily_job(session)
    if status is not None:
        worker_job(session, status, **stats)
    notices = collect_digest(session, user, now=NOW).notices
    assert any(n.startswith(expected) for n in notices), notices


def test_send_once_per_day(session: Session) -> None:
    user = setup(session)
    daily_job(session)
    analyse(session, user, add_video(session, "a"))
    notifier = FakeNotifier()

    first = send_digest(session, user, notifier, now=NOW)
    second = send_digest(session, user, notifier, now=NOW)

    assert (first.status, second.status) == ("sent", "already_sent")
    assert len(notifier.pushes) == 1
    row = session.scalars(select(Digest)).one()
    assert row.status is DigestStatus.SENT and row.sent_at == NOW
    assert row.retry_key == notifier.pushes[0][1]
    assert "summary a" in (row.message_text or "")
    assert (
        session.scalars(select(JobRun.status).where(JobRun.job_name == "send_digest")).one()
        is JobStatus.SUCCEEDED
    )


def test_failed_push_is_recorded_and_retried_with_the_same_key(session: Session) -> None:
    user = setup(session)
    failing = FakeNotifier(error=NotifyError("HTTP 429: monthly limit"))

    outcome = send_digest(session, user, failing, now=NOW)
    assert outcome.status == "failed" and "429" in (outcome.error or "")
    row = session.scalars(select(Digest)).one()
    assert row.status is DigestStatus.FAILED

    retry = FakeNotifier(delivered=False)  # LINE says this key was already accepted earlier
    assert send_digest(session, user, retry, now=NOW).status == "sent"
    assert retry.pushes[0][1] == failing.pushes[0][1]


@pytest.fixture
def cli_db(engine, test_database_url, tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("LINE_CHANNEL_ACCESS_TOKEN", "token")
    monkeypatch.setenv("LINE_USER_ID", "U123")
    with Session(engine) as db:
        user = setup(db)
        daily_job(db)
        analyse(db, user, add_video(db, "a"))
        db.commit()
    yield engine
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE users, channels, job_runs RESTART IDENTITY CASCADE"))


def test_cli_digest_dry_run_prints_and_saves_nothing(cli_db: Engine) -> None:
    result = CliRunner().invoke(cli.app, ["digest", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "----- message 1/" in result.output and "summary a" in result.output
    with Session(cli_db) as db:
        assert db.scalars(select(Digest)).all() == []


@respx.mock
def test_cli_digest_sends_then_skips(cli_db: Engine) -> None:
    route = respx.post(PUSH_URL).respond(200, json={})
    runner = CliRunner()
    assert "Sent digest" in runner.invoke(cli.app, ["digest"]).output
    assert "already sent" in runner.invoke(cli.app, ["digest"]).output
    assert route.call_count == 1


@respx.mock
def test_cli_digest_reports_failure(cli_db: Engine) -> None:
    respx.post(PUSH_URL).respond(401, json={"message": "Authentication failed"})
    result = CliRunner().invoke(cli.app, ["digest"])
    assert result.exit_code == 1
    assert "could not be sent" in result.output


def test_cli_digest_requires_line_settings(cli_db: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LINE_CHANNEL_ACCESS_TOKEN")
    result = CliRunner().invoke(cli.app, ["digest"])
    assert result.exit_code == 1
    assert "LINE_CHANNEL_ACCESS_TOKEN" in result.output


def test_feed_failures_become_a_notice(session: Session) -> None:
    user = setup(session)
    daily_job(session, feed_errors={"LangChain": "FeedNotFoundError", "statquest": "FeedNotFoundError"})
    notices = collect_digest(session, user, now=NOW).notices
    assert any("讀取 2 個頻道的 RSS 失敗" in n for n in notices)
