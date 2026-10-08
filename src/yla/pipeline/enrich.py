"""Add length and live status to new videos, skip Shorts, and hold back unfinished streams.

Optional: without a YouTube API key (or if the API fails) this step is skipped and videos are
treated as ordinary videos. Runs in both the local worker (so no caption requests are spent on
Shorts) and the cloud pipeline; it only touches videos without metadata or still waiting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from yla.db.models import LiveStatus, Subscription, User, Video, VideoStatus
from yla.youtube.data_api import MetadataError, MetadataProvider

logger = logging.getLogger(__name__)

# YouTube Shorts can be up to 3 minutes long.
SHORTS_MAX_SECONDS = 180


@dataclass
class EnrichReport:
    checked: int = 0
    shorts: list[str] = field(default_factory=list)
    waiting: list[str] = field(default_factory=list)  # upcoming premiere or live stream in progress
    released: list[str] = field(default_factory=list)  # stream finished: now a normal video
    missing: list[str] = field(default_factory=list)  # private or deleted
    expired: list[str] = field(default_factory=list)  # waited longer than defer_max_days
    error: str | None = None


def enrich_videos(session: Session, user: User, provider: MetadataProvider, *, now: datetime) -> EnrichReport:
    report = EnrichReport()
    videos = list(
        session.scalars(
            select(Video)
            .join(
                Subscription,
                (Subscription.channel_id == Video.channel_id)
                & (Subscription.user_id == user.id)
                & Subscription.enabled.is_(True),
            )
            .where(
                or_(
                    Video.status == VideoStatus.WAITING,
                    Video.duration_seconds.is_(None)
                    & Video.status.in_([VideoStatus.PENDING, VideoStatus.DEFERRED]),
                )
            )
        )
    )
    if not videos:
        return report

    video_ids = [v.youtube_video_id for v in videos]
    session.commit()  # no transaction may stay open while waiting on the API
    try:
        found = provider.fetch(video_ids)
    except MetadataError as exc:
        logger.warning("YouTube Data API unavailable, treating videos as ordinary: %s", exc)
        report.error = str(exc)[:300]
        return report

    report.checked = len(videos)
    expiry = now - timedelta(days=user.defer_max_days)
    for video in videos:
        meta = found.get(video.youtube_video_id)
        if meta is None:
            video.status, video.last_error = (
                VideoStatus.UNAVAILABLE,
                "not returned by the Data API (private or deleted)",
            )
            report.missing.append(video.youtube_video_id)
            continue

        video.live_status = meta.live_status
        if meta.live_status in (LiveStatus.UPCOMING, LiveStatus.LIVE):
            if video.discovered_at < expiry:
                video.status, video.last_error = (
                    VideoStatus.FAILED,
                    "stream did not finish within defer_max_days",
                )
                report.expired.append(video.youtube_video_id)
            else:
                video.status = VideoStatus.WAITING
                report.waiting.append(video.youtube_video_id)
            continue

        video.duration_seconds = meta.duration_seconds
        video.is_short = meta.duration_seconds <= SHORTS_MAX_SECONDS
        if video.is_short and not user.include_shorts:
            video.status = VideoStatus.SKIPPED_SHORT
            report.shorts.append(video.youtube_video_id)
        elif video.status is VideoStatus.WAITING:
            video.status = VideoStatus.PENDING
            report.released.append(video.youtube_video_id)
    session.flush()
    return report
