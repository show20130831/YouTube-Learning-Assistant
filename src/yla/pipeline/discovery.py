"""Detect new videos from the RSS feeds of the user's channels.

Runs on both the local worker and Modal; inserts are idempotent (ON CONFLICT DO NOTHING), so
whichever runs first wins and the other is a no-op.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from yla.db.models import Channel, Subscription, Video, VideoStatus
from yla.youtube.rss import FeedFetcher, FeedNotFoundError, FeedParseError

logger = logging.getLogger(__name__)

TITLE_MAX = 300


@dataclass
class DiscoveryReport:
    channels_checked: int = 0
    new_videos: list[str] = field(default_factory=list)  # YouTube IDs queued as pending
    backfilled: int = 0  # already in the feed on a channel's first read; never analysed
    failed_channels: dict[str, str] = field(default_factory=dict)
    skipped_without_id: list[str] = field(default_factory=list)


def discover_videos(
    session: Session, user_id: int, fetcher: FeedFetcher, *, now: datetime
) -> DiscoveryReport:
    report = DiscoveryReport()
    rows = session.execute(
        select(Channel, Subscription)
        .join(Subscription, Subscription.channel_id == Channel.id)
        .where(Subscription.user_id == user_id, Subscription.enabled.is_(True))
        .order_by(Channel.id)
    ).all()

    for channel, sub in rows:
        if channel.youtube_channel_id is None:
            report.skipped_without_id.append(channel.handle)
            continue
        try:
            feed = fetcher.fetch(channel.youtube_channel_id)
        except (httpx.HTTPError, FeedNotFoundError, FeedParseError) as exc:
            # Leave last_checked_at untouched so a first read keeps its backfill semantics.
            logger.warning("feed failed for %s: %s", channel.handle, exc)
            report.failed_channels[channel.handle] = f"{type(exc).__name__}: {exc}"
            continue

        report.channels_checked += 1
        if channel.title is None and feed.channel_title:
            channel.title = feed.channel_title

        first_read = sub.last_checked_at is None
        status = VideoStatus.SKIPPED_BACKFILL if first_read else VideoStatus.PENDING
        if feed.entries:
            stmt = (
                insert(Video)
                .values(
                    [
                        {
                            "youtube_video_id": e.video_id,
                            "channel_id": channel.id,
                            "title": e.title[:TITLE_MAX],
                            "description": e.description,
                            "published_at": e.published_at,
                            "status": status,
                            "attempts": 0,
                        }
                        for e in feed.entries
                    ]
                )
                .on_conflict_do_nothing(index_elements=[Video.youtube_video_id])
                .returning(Video.youtube_video_id)
            )
            inserted = list(session.scalars(stmt))
            if first_read:
                report.backfilled += len(inserted)
            else:
                report.new_videos.extend(inserted)
        sub.last_checked_at = now

    session.flush()
    return report
