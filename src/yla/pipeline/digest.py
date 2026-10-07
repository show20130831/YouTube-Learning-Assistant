"""The daily digest job (Modal cron at 12:00, or ``yla digest`` locally).

Collects everything analysed since the last digest that was actually sent (at most 3 days
back, so a failed day is not lost), formats it and pushes it to LINE once per local day:
- a digest already marked ``sent`` for today is never pushed again;
- each day's digest has a stored LINE retry key, so even a crash between pushing and recording
  "sent" cannot produce a duplicate message.
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import ColumnElement, func, select
from sqlalchemy.orm import Session

from yla.db.models import (
    Analysis,
    Channel,
    Digest,
    DigestStatus,
    JobRun,
    JobStatus,
    Subscription,
    User,
    UserVideoRelevance,
    Video,
    VideoStatus,
)
from yla.notify.formatter import DigestData, DigestVideo, build_messages
from yla.notify.line import Notifier, NotifyError
from yla.pipeline.daily import JOB_NAME as DAILY_JOB_NAME
from yla.worker.local_worker import JOB_NAME as WORKER_JOB_NAME

logger = logging.getLogger(__name__)

JOB_NAME = "send_digest"
MAX_LOOKBACK = timedelta(days=3)
TREND_DAYS = 7
TREND_TOP = 5
MESSAGE_SEPARATOR = "\n\n=====\n\n"


@dataclass(frozen=True)
class DigestOutcome:
    status: str  # "sent" | "already_sent" | "failed"
    day: date
    messages: list[str]
    error: str | None = None


def _local_midnight(now: datetime, tz: ZoneInfo) -> datetime:
    return datetime.combine(now.astimezone(tz).date(), time(), tzinfo=tz)


def _subscribed(user: User) -> ColumnElement[bool]:
    return (Subscription.channel_id == Video.channel_id) & (Subscription.user_id == user.id)


def collect_digest(session: Session, user: User, *, now: datetime) -> DigestData:
    tz = ZoneInfo(user.timezone)
    today = now.astimezone(tz).date()
    midnight = _local_midnight(now, tz)
    last_sent = session.scalar(
        select(func.max(Digest.sent_at)).where(
            Digest.user_id == user.id, Digest.status == DigestStatus.SENT, Digest.digest_date < today
        )
    )
    since = max(last_sent or midnight, now - MAX_LOOKBACK)

    # Latest analysis per video in the window.
    latest = (
        select(Analysis.video_id, func.max(Analysis.id).label("analysis_id"))
        .where(Analysis.created_at >= since)
        .group_by(Analysis.video_id)
        .subquery()
    )
    rows = session.execute(
        select(Analysis, Video, Channel, UserVideoRelevance)
        .join(latest, latest.c.analysis_id == Analysis.id)
        .join(Video, Video.id == Analysis.video_id)
        .join(Channel, Channel.id == Video.channel_id)
        .join(Subscription, _subscribed(user))
        .outerjoin(
            UserVideoRelevance,
            (UserVideoRelevance.video_id == Video.id) & (UserVideoRelevance.user_id == user.id),
        )
    ).all()

    videos: list[DigestVideo] = []
    for analysis, video, channel, relevance in rows:
        earlier_level = session.scalar(
            select(func.max(Analysis.confidence_level)).where(
                Analysis.video_id == video.id, Analysis.id < analysis.id
            )
        )
        videos.append(
            DigestVideo(
                title=video.title,
                channel=channel.title or channel.handle,
                url=f"https://youtube.com/watch?v={video.youtube_video_id}",
                source_type=analysis.source_type,
                confidence_level=analysis.confidence_level,
                one_line_summary=analysis.one_line_summary,
                key_points=list(analysis.key_points),
                keywords=list(analysis.keywords),
                matched_topics=list(relevance.matched_topics) if relevance else [],
                relevance_score=relevance.relevance_score if relevance else 0,
                upgraded=earlier_level is not None and earlier_level < analysis.confidence_level,
            )
        )

    recent_videos = (
        select(Video, Channel)
        .join(Channel, Channel.id == Video.channel_id)
        .join(Subscription, _subscribed(user))
        .where(Video.discovered_at >= since)
    )
    discovered = session.execute(recent_videos).all()
    not_new = (VideoStatus.SKIPPED_BACKFILL, VideoStatus.SKIPPED_SHORT)
    new_videos = sum(1 for v, _ in discovered if v.status not in not_new)
    shorts_skipped = sum(1 for v, _ in discovered if v.status is VideoStatus.SKIPPED_SHORT)
    unavailable = [
        f"{v.title}｜{c.title or c.handle}" for v, c in discovered if v.status is VideoStatus.UNAVAILABLE
    ]
    deferred = (
        session.scalar(
            select(func.count(Video.id))
            .join(Subscription, _subscribed(user))
            .where(Video.status == VideoStatus.DEFERRED, Subscription.enabled.is_(True))
        )
        or 0
    )
    channels_tracked = (
        session.scalar(
            select(func.count()).where(Subscription.user_id == user.id, Subscription.enabled.is_(True))
        )
        or 0
    )

    worker_info, worker_warnings = _worker_status(session, midnight=midnight, tz=tz)
    return DigestData(
        day=today,
        channels_tracked=channels_tracked,
        new_videos=new_videos,
        shorts_skipped=shorts_skipped,
        videos=videos,
        unavailable=unavailable,
        deferred=deferred,
        trends=_trends(session, user, now=now),
        status=worker_info,
        notices=[*worker_warnings, *_notices(session, videos, deferred, midnight=midnight)],
    )


def _trends(session: Session, user: User, *, now: datetime) -> list[tuple[str, int]]:
    """How many videos matched each topic in the last 7 days (simple v1; Phase 2 adds keywords)."""
    recent = select(Analysis.video_id).where(Analysis.created_at >= now - timedelta(days=TREND_DAYS))
    rows = session.scalars(
        select(UserVideoRelevance.matched_topics).where(
            UserVideoRelevance.user_id == user.id, UserVideoRelevance.video_id.in_(recent)
        )
    )
    counts = Counter(topic for topics in rows for topic in set(topics))
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:TREND_TOP]


def _notices(session: Session, videos: list[DigestVideo], deferred: int, *, midnight: datetime) -> list[str]:
    notices: list[str] = []
    job = session.scalars(
        select(JobRun)
        .where(JobRun.job_name == DAILY_JOB_NAME, JobRun.started_at >= midnight)
        .order_by(JobRun.id.desc())
        .limit(1)
    ).first()
    if job is None:
        notices.append("⚠️ 今日分析流程沒有執行，請檢查排程。")
    elif job.status is JobStatus.FAILED:
        notices.append(f"⚠️ 今日分析流程失敗：{(job.error or '')[:100]}")
    else:
        if job.stats.get("quota_exhausted"):
            notices.append(f"⚠️ OpenRouter 免費額度已用完，{deferred} 部影片延到明天。")
        if feed_errors := job.stats.get("feed_errors"):
            notices.append(
                f"⚠️ 今日分析流程讀取 {len(feed_errors)} 個頻道的 RSS 失敗"
                "（YouTube 端暫時性錯誤），下次執行會再試。"
            )

    return notices


def _worker_status(session: Session, *, midnight: datetime, tz: ZoneInfo) -> tuple[list[str], list[str]]:
    """Today's local worker run as (info lines, warnings), so problems at home show up on LINE."""
    job = session.scalars(
        select(JobRun)
        .where(JobRun.job_name == WORKER_JOB_NAME, JobRun.started_at >= midnight)
        .order_by(JobRun.id.desc())
        .limit(1)
    ).first()
    if job is None:
        return [], ["⚠️ 本機 worker 今天沒有執行（電腦可能沒開機），今天的影片只能用標題與描述摘要。"]
    if job.status is JobStatus.FAILED:
        return [], [f"⚠️ 本機 worker 執行失敗：{(job.error or '')[:100]}"]
    if job.status is JobStatus.RUNNING:
        return [], ["⚠️ 本機 worker 尚未執行完成（可能中途中斷）。"]

    stats = job.stats
    warnings: list[str] = []
    if stats.get("blocked"):
        warnings.append("⚠️ 本機 worker 的字幕請求被 YouTube 封鎖，今天的影片只能用標題與描述摘要。")
    feed_errors = len(stats.get("feed_errors") or {})
    feeds = f"，RSS 失敗 {feed_errors} 個頻道" if feed_errors else ""
    when = job.started_at.astimezone(tz).strftime("%H:%M")
    info = f"💻 本機 worker：{when} 完成（字幕 {stats.get('saved', 0)}/{stats.get('attempted', 0)}{feeds}）"
    return [info], warnings


def send_digest(session: Session, user: User, notifier: Notifier, *, now: datetime) -> DigestOutcome:
    day = now.astimezone(ZoneInfo(user.timezone)).date()
    digest = session.scalars(
        select(Digest).where(Digest.user_id == user.id, Digest.digest_date == day)
    ).first()
    if digest is not None and digest.status is DigestStatus.SENT:
        return DigestOutcome(status="already_sent", day=day, messages=[])

    messages = build_messages(collect_digest(session, user, now=now))
    if digest is None:
        digest = Digest(user_id=user.id, digest_date=day)
        session.add(digest)
    digest.retry_key = digest.retry_key or str(uuid.uuid4())
    digest.status, digest.message_text, digest.error = (
        DigestStatus.PENDING,
        MESSAGE_SEPARATOR.join(messages),
        None,
    )
    job = JobRun(job_name=JOB_NAME, status=JobStatus.RUNNING, stats={"messages": len(messages)})
    session.add(job)
    session.commit()  # the retry key must be stored before pushing

    try:
        delivered = notifier.push(messages, retry_key=digest.retry_key)
    except NotifyError as exc:
        digest.status, digest.error = DigestStatus.FAILED, str(exc)[:2000]
        job.status, job.error, job.finished_at = JobStatus.FAILED, digest.error, now
        session.commit()
        logger.error("digest push failed: %s", exc)
        return DigestOutcome(status="failed", day=day, messages=messages, error=digest.error)

    digest.status, digest.sent_at = DigestStatus.SENT, now
    job.status, job.finished_at = JobStatus.SUCCEEDED, now
    job.stats = {**job.stats, "delivered": delivered}
    session.commit()
    return DigestOutcome(status="sent", day=day, messages=messages)
