from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from yla.config import AppSettings
from yla.db.models import (
    Analysis,
    Channel,
    JobRun,
    JobStatus,
    SourceType,
    Subscription,
    User,
    Video,
    VideoStatus,
)
from yla.ops.alerts import trend_warnings
from yla.sync import sync_config

pytestmark = pytest.mark.db

NOW = datetime.now(UTC)
LC = "UCC-lyoTfSrcJzA1ab3APAgw"
SQ = "UCtYLUTtgS3k1Fg4y5tAhLbw"


def setup(session: Session) -> User:
    channels = [{"handle": "LangChain", "channel_id": LC}, {"handle": "statquest", "channel_id": SQ}]
    sync_config(session, AppSettings.model_validate({"channels": channels}))
    for sub in session.scalars(select(Subscription)):
        sub.last_checked_at = NOW - timedelta(hours=3)
    session.flush()
    return session.scalars(select(User)).one()


def job(session: Session, name: str, status: JobStatus = JobStatus.SUCCEEDED, **stats: Any) -> None:
    session.add(JobRun(job_name=name, status=status, stats=stats))
    session.flush()


def warnings(session: Session, user: User) -> list[str]:
    return trend_warnings(session, user, now=NOW)


def test_healthy_system_has_no_warnings(session: Session) -> None:
    user = setup(session)
    job(session, "daily_pipeline", feed_errors={"LangChain": "x", "statquest": "x"})  # routine cloud failures
    assert warnings(session, user) == []


def test_channel_not_read_by_any_job_for_two_days(session: Session) -> None:
    user = setup(session)
    sub = session.scalars(
        select(Subscription)
        .join(Channel, Channel.id == Subscription.channel_id)
        .where(Channel.handle == "statquest")
    ).one()
    sub.last_checked_at = NOW - timedelta(days=3, hours=1)
    session.flush()
    assert warnings(session, user) == ["⚠️ statquest 已 3 天無法讀取 RSS，請確認頻道 ID 是否正確。"]


def test_never_read_channel_counts_from_when_it_was_added(session: Session) -> None:
    user = setup(session)
    sub = session.scalars(select(Subscription)).first()
    assert sub is not None
    sub.last_checked_at, sub.subscribed_at = None, NOW - timedelta(days=5)
    session.flush()
    assert any("5 天無法讀取" in w for w in warnings(session, user))


def test_primary_model_failing_in_three_runs_in_a_row(session: Session) -> None:
    user = setup(session)
    for _ in range(3):
        job(session, "daily_pipeline", analysis_attempts=2, primary_failures=1, primary_model="m1")
    assert any(w.startswith("⚠️ m1 最近 3 次執行都有一半以上失敗") for w in warnings(session, user))


def test_primary_model_warning_needs_three_bad_runs(session: Session) -> None:
    user = setup(session)
    job(session, "daily_pipeline", analysis_attempts=4, primary_failures=0, primary_model="m1")
    job(session, "daily_pipeline", analysis_attempts=0)  # nothing analysed: ignored
    for _ in range(2):
        job(session, "daily_pipeline", analysis_attempts=2, primary_failures=2, primary_model="m1")
    assert warnings(session, user) == []


def test_caption_errors_in_two_worker_runs(session: Session) -> None:
    user = setup(session)
    job(session, "local_worker", errors=1, no_captions=0)
    job(session, "local_worker", errors=2, no_captions=3)
    assert any("連續 2 次抓字幕出錯" in w for w in warnings(session, user))


@pytest.mark.parametrize(
    "second",
    [
        {"errors": 0, "no_captions": 4},  # videos without captions are not an error
        {"errors": 1, "blocked": True},  # blocking has its own warning
    ],
)
def test_no_caption_warning_for_missing_captions_or_blocks(session: Session, second: dict[str, Any]) -> None:
    user = setup(session)
    job(session, "local_worker", errors=1)
    job(session, "local_worker", **second)
    assert warnings(session, user) == []


def add_analyses(session: Session, levels: list[int]) -> None:
    channel_id = session.scalars(select(Channel.id)).first()
    for i, level in enumerate(levels):
        video = Video(
            youtube_video_id=f"v{i}",
            channel_id=channel_id,
            title="t",
            description="",
            published_at=NOW,
            status=VideoStatus.ANALYZED,
        )
        session.add(video)
        session.flush()
        source = SourceType.TITLE_DESCRIPTION if level == 1 else SourceType.AUTO_CAPTION
        session.add(
            Analysis(
                video_id=video.id,
                model="m",
                prompt_version="p",
                source_type=source,
                confidence_level=level,
                confidence_stars="⭐" * level,
                one_line_summary="s",
                key_points=[],
                key_concepts=[],
                keywords=[],
                raw_response={},
            )
        )
    session.flush()


def test_mostly_topic_guesses_this_week(session: Session) -> None:
    user = setup(session)
    add_analyses(session, [1, 1, 1, 2])
    assert warnings(session, user) == ["⚠️ 近 7 天 3/4 部摘要沒有字幕（⭐），請確認本機 worker 是否每天執行。"]


@pytest.mark.parametrize("levels", [[1, 1, 2, 2], [1, 1, 1]])  # half is fine; too few to judge
def test_no_quality_warning_at_half_or_with_few_videos(session: Session, levels: list[int]) -> None:
    user = setup(session)
    add_analyses(session, levels)
    assert warnings(session, user) == []


def test_trend_warnings_appear_in_the_digest(session: Session) -> None:
    from yla.pipeline.digest import collect_digest

    user = setup(session)
    job(session, "local_worker", errors=1)
    job(session, "local_worker", errors=1)
    assert any("連續 2 次抓字幕出錯" in n for n in collect_digest(session, user, now=NOW).notices)
