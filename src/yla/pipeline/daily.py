"""The daily analysis job (Modal cron at 11:00, or ``yla run`` locally).

1. discover new videos from RSS (idempotent; the local worker may already have done it);
2. give up on videos still unanalysed after ``defer_max_days``;
3. grade each candidate's content; videos without enough text become ``unavailable`` without
   using any of the daily limit;
4. rank the rest by topic matches in title + description (no LLM requests spent on ranking);
5. analyse up to the daily limit, minus what was already analysed today (re-runs are safe),
   and defer the rest to tomorrow;
6. record a ``job_runs`` row with the counts and the number of LLM requests.

If every model is rate-limited, the free quota is assumed spent: analysis stops and the
remaining videos are deferred instead of failing.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from yla.analysis.analyzer import AnalysisFailed, Analyzer, store_analysis
from yla.analysis.topics import TopicMatcher
from yla.config import AppSettings
from yla.content.tiering import ContentTier, choose_content
from yla.db.models import Analysis, Channel, JobRun, JobStatus, Subscription, User, Video, VideoStatus
from yla.pipeline.discovery import DiscoveryReport, discover_videos
from yla.pipeline.enrich import enrich_videos
from yla.youtube.data_api import MetadataProvider
from yla.youtube.rss import FeedFetcher
from yla.youtube.transcripts import DatabaseTranscriptProvider

logger = logging.getLogger(__name__)

JOB_NAME = "daily_pipeline"
MAX_ATTEMPTS = 3
CANDIDATE_STATUSES = (VideoStatus.PENDING, VideoStatus.DEFERRED)


@dataclass
class DailyReport:
    run_id: str
    discovered: int = 0
    analyzed: list[str] = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)
    deferred: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    expired: list[str] = field(default_factory=list)
    llm_calls: int = 0
    quota_exhausted: bool = False
    feed_errors: dict[str, str] = field(default_factory=dict)
    shorts_skipped: int = 0
    waiting: int = 0
    metadata_error: str | None = None

    def stats(self) -> dict[str, object]:
        data = asdict(self)
        return {k: (len(v) if isinstance(v, list) else v) for k, v in data.items()}


@dataclass
class _Candidate:
    video: Video
    channel_name: str
    tier: ContentTier
    topic_hits: int


def analyzed_today(session: Session, user: User, now: datetime) -> int:
    """Analyses created since local midnight in the user's timezone (counts toward the limit)."""
    local_now = now.astimezone(ZoneInfo(user.timezone))
    midnight = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
    stmt = (
        select(func.count(Analysis.id))
        .join(Video, Video.id == Analysis.video_id)
        .join(Subscription, (Subscription.channel_id == Video.channel_id) & (Subscription.user_id == user.id))
        .where(Analysis.created_at >= midnight)
    )
    return session.scalar(stmt) or 0


def _candidates(session: Session, user: User) -> list[tuple[Video, Channel]]:
    stmt = (
        select(Video, Channel)
        .join(Channel, Channel.id == Video.channel_id)
        .join(
            Subscription,
            (Subscription.channel_id == Channel.id)
            & (Subscription.user_id == user.id)
            & Subscription.enabled.is_(True),
        )
        .where(Video.status.in_(CANDIDATE_STATUSES))
        .order_by(Video.published_at.desc())
    )
    return [(video, channel) for video, channel in session.execute(stmt)]


def run_daily_pipeline(
    session: Session,
    user: User,
    settings: AppSettings,
    analyzer: Analyzer,
    *,
    now: datetime,
    feed_fetcher: FeedFetcher | None = None,
    metadata: MetadataProvider | None = None,
) -> DailyReport:
    report = DailyReport(run_id=uuid.uuid4().hex[:8])
    job = JobRun(job_name=JOB_NAME, status=JobStatus.RUNNING, stats={"run_id": report.run_id})
    session.add(job)
    session.commit()
    logger.info("daily pipeline %s started", report.run_id)

    try:
        if feed_fetcher is not None:
            discovery: DiscoveryReport = discover_videos(session, user.id, feed_fetcher, now=now)
            report.discovered = len(discovery.new_videos)
            report.feed_errors = discovery.failed_channels
            session.commit()
        if metadata is not None:
            enriched = enrich_videos(session, user, metadata, now=now)
            report.shorts_skipped, report.waiting = len(enriched.shorts), len(enriched.waiting)
            report.metadata_error = enriched.error
            session.commit()

        _process(session, user, settings, analyzer, now=now, report=report)
    except Exception as exc:
        session.rollback()
        job.status, job.error = JobStatus.FAILED, f"{type(exc).__name__}: {exc}"[:2000]
        job.finished_at, job.stats = now, report.stats()
        session.commit()
        raise

    job.status, job.finished_at, job.stats = JobStatus.SUCCEEDED, now, report.stats()
    session.commit()
    logger.info("daily pipeline %s finished: %s", report.run_id, job.stats)
    return report


def _process(
    session: Session,
    user: User,
    settings: AppSettings,
    analyzer: Analyzer,
    *,
    now: datetime,
    report: DailyReport,
) -> None:
    matcher = TopicMatcher(settings.topics)
    transcripts = DatabaseTranscriptProvider(session)
    expiry = now - timedelta(days=user.defer_max_days)
    ready: list[_Candidate] = []

    for video, channel in _candidates(session, user):
        if video.discovered_at < expiry:
            video.status, video.last_error = VideoStatus.FAILED, "not analysed within defer_max_days"
            report.expired.append(video.youtube_video_id)
            continue
        tier = choose_content(
            transcripts.get(video.id),
            video.title,
            video.description,
            min_description_chars=settings.content.min_description_chars,
        )
        if not tier.available:
            video.status = VideoStatus.UNAVAILABLE
            report.unavailable.append(video.youtube_video_id)
            continue
        hits = len(matcher.from_text(f"{video.title}\n{video.description}"))
        ready.append(_Candidate(video, channel.title or channel.handle, tier, hits))
    session.commit()

    # Most topic matches first; ties keep the newest-first order (stable sort).
    ready.sort(key=lambda c: c.topic_hits, reverse=True)
    budget = max(0, user.daily_video_limit - analyzed_today(session, user, now))

    for candidate in ready:
        video = candidate.video
        if budget <= 0 or report.quota_exhausted:
            _defer(video, now, report)
            continue
        title = video.title
        session.commit()  # no transaction may stay open during a (possibly minutes-long) LLM call
        try:
            outcome = analyzer.analyze(title=title, channel=candidate.channel_name, tier=candidate.tier)
        except AnalysisFailed as exc:
            report.llm_calls += exc.calls
            video.last_error = str(exc)[:2000]
            if exc.rate_limited:
                report.quota_exhausted = True
                _defer(video, now, report)
            else:
                video.attempts += 1
                if video.attempts >= MAX_ATTEMPTS:
                    video.status = VideoStatus.FAILED
                    report.failed.append(video.youtube_video_id)
                else:
                    _defer(video, now, report)
            session.commit()
            continue

        report.llm_calls += outcome.calls
        store_analysis(session, video=video, user_id=user.id, outcome=outcome)
        report.analyzed.append(video.youtube_video_id)
        budget -= 1
        session.commit()


def _defer(video: Video, now: datetime, report: DailyReport) -> None:
    video.status = VideoStatus.DEFERRED
    if video.deferred_since is None:
        video.deferred_since = now
    report.deferred.append(video.youtube_video_id)
