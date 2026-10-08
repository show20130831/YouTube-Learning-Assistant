"""Local worker: runs on the user's own computer (home network) before the Modal pipeline.

1. discover new videos from RSS (same idempotent step Modal runs, so either can go first);
2. fetch captions for recent videos that have none yet, and store them.

It never calls the LLM or LINE. If YouTube blocks a request it stops immediately: retrying
from a blocked IP only makes things worse, and videos simply fall back to title + description.
"""

from __future__ import annotations

import logging
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from yla.db.models import JobRun, JobStatus, Subscription, Transcript, User, Video, VideoStatus
from yla.pipeline.discovery import DiscoveryReport, discover_videos
from yla.pipeline.enrich import enrich_videos
from yla.youtube.data_api import MetadataProvider
from yla.youtube.rss import FeedFetcher
from yla.youtube.transcripts import TranscriptProvider, TranscriptSink, TranscriptStatus

logger = logging.getLogger(__name__)

# Statuses that may still need (or benefit from) captions. ANALYZED without a transcript means
# the summary was a ⭐ title/description guess: captions found later upgrade it.
# Unfinished streams (WAITING) are left out: they have no captions yet.
NEEDS_TRANSCRIPT = (VideoStatus.PENDING, VideoStatus.DEFERRED, VideoStatus.ANALYZED)


@dataclass
class TranscriptFetchReport:
    attempted: int = 0
    saved: dict[str, TranscriptStatus] = field(default_factory=dict)
    no_captions: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    blocked: bool = False


def select_transcript_candidates(
    session: Session, user_id: int, *, now: datetime, lookback_days: int, limit: int
) -> list[Video]:
    has_transcript = exists().where(Transcript.video_id == Video.id)
    stmt = (
        select(Video)
        .join(
            Subscription,
            (Subscription.channel_id == Video.channel_id)
            & (Subscription.user_id == user_id)
            & Subscription.enabled.is_(True),
        )
        .where(
            Video.discovered_at >= now - timedelta(days=lookback_days),
            Video.status.in_(NEEDS_TRANSCRIPT),
            ~has_transcript,
        )
        .order_by(Video.published_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def fetch_transcripts(
    session: Session,
    user_id: int,
    provider: TranscriptProvider,
    sink: TranscriptSink,
    *,
    now: datetime,
    limit: int,
    lookback_days: int = 3,
    delay_seconds: float = 1.5,
    sleep: Callable[[float], None] = time.sleep,
) -> TranscriptFetchReport:
    """Fetch and store captions, committing after each video so progress survives a crash."""
    report = TranscriptFetchReport()
    candidates = [
        (v.id, v.youtube_video_id)
        for v in select_transcript_candidates(
            session, user_id, now=now, lookback_days=lookback_days, limit=limit
        )
    ]
    session.commit()  # no transaction may stay open while waiting on YouTube

    for index, (video_pk, video_id) in enumerate(candidates):
        if index:
            sleep(delay_seconds)  # stay well below anything resembling scraping
        report.attempted += 1
        result = provider.fetch(video_id)

        if result.status is TranscriptStatus.BLOCKED:
            logger.warning("YouTube blocked caption requests (%s); stopping", result.error)
            report.blocked = True
            report.errors[video_id] = result.error or "blocked"
            break
        if result.has_text:
            sink.save(video_pk, result)
            session.commit()
            report.saved[video_id] = result.status
        elif result.status is TranscriptStatus.NONE:
            report.no_captions.append(video_id)
        else:
            report.errors[video_id] = result.error or result.status.value
    return report


JOB_NAME = "local_worker"


@dataclass
class WorkerReport:
    discovery: DiscoveryReport | None
    transcripts: TranscriptFetchReport


def run_local_worker(
    session: Session,
    user: User,
    provider: TranscriptProvider,
    sink: TranscriptSink,
    *,
    now: datetime,
    feed_fetcher: FeedFetcher | None = None,
    metadata: MetadataProvider | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> WorkerReport:
    """Discover + fetch captions, recorded as a ``local_worker`` job so the cloud side (and the
    LINE digest) can tell whether the home computer did its part today."""
    job = JobRun(job_name=JOB_NAME, status=JobStatus.RUNNING, stats={"host": socket.gethostname()})
    session.add(job)
    session.commit()

    discovery: DiscoveryReport | None = None
    try:
        if feed_fetcher is not None:
            discovery = discover_videos(session, user.id, feed_fetcher, now=now)
            session.commit()
        enriched = None
        if metadata is not None:  # before captions, so none are fetched for Shorts
            enriched = enrich_videos(session, user, metadata, now=now)
            session.commit()
        transcripts = fetch_transcripts(
            session, user.id, provider, sink, now=now, limit=user.daily_video_limit * 2, sleep=sleep
        )
    except Exception as exc:
        session.rollback()
        job.status, job.error = JobStatus.FAILED, f"{type(exc).__name__}: {exc}"[:2000]
        job.finished_at = now
        session.commit()
        raise

    saved = list(transcripts.saved.values())
    job.stats = {
        **job.stats,
        "new_videos": len(discovery.new_videos) if discovery else None,
        "feed_errors": discovery.failed_channels if discovery else {},
        "attempted": transcripts.attempted,
        "saved": len(saved),
        "manual": saved.count(TranscriptStatus.MANUAL),
        "auto": saved.count(TranscriptStatus.GENERATED),
        "no_captions": len(transcripts.no_captions),
        "errors": len(transcripts.errors),
        "blocked": transcripts.blocked,
        "shorts_skipped": len(enriched.shorts) if enriched else None,
        "metadata_error": enriched.error if enriched else None,
    }
    job.status, job.finished_at = JobStatus.SUCCEEDED, now
    session.commit()
    return WorkerReport(discovery=discovery, transcripts=transcripts)
