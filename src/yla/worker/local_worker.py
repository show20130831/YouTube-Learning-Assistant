"""Local worker: runs on the user's own computer (home network) before the Modal pipeline.

1. discover new videos from RSS (same idempotent step Modal runs, so either can go first);
2. fetch captions for recent videos that have none yet, and store them.

It never calls the LLM or LINE. If YouTube blocks a request it stops immediately: retrying
from a blocked IP only makes things worse, and videos simply fall back to title + description.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from yla.db.models import Subscription, Transcript, Video, VideoStatus
from yla.youtube.transcripts import TranscriptProvider, TranscriptSink, TranscriptStatus

logger = logging.getLogger(__name__)

# Statuses that may still need (or benefit from) captions. ANALYZED without a transcript means
# the summary was a ⭐ title/description guess: captions found later upgrade it.
NEEDS_TRANSCRIPT = (VideoStatus.PENDING, VideoStatus.DEFERRED, VideoStatus.WAITING, VideoStatus.ANALYZED)


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
    videos = select_transcript_candidates(session, user_id, now=now, lookback_days=lookback_days, limit=limit)

    for index, video in enumerate(videos):
        if index:
            sleep(delay_seconds)  # stay well below anything resembling scraping
        report.attempted += 1
        result = provider.fetch(video.youtube_video_id)

        if result.status is TranscriptStatus.BLOCKED:
            logger.warning("YouTube blocked caption requests (%s); stopping", result.error)
            report.blocked = True
            report.errors[video.youtube_video_id] = result.error or "blocked"
            break
        if result.has_text:
            sink.save(video.id, result)
            session.commit()
            report.saved[video.youtube_video_id] = result.status
        elif result.status is TranscriptStatus.NONE:
            report.no_captions.append(video.youtube_video_id)
        else:
            report.errors[video.youtube_video_id] = result.error or result.status.value
    return report
