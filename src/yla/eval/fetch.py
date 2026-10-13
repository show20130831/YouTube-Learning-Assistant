"""Fill the local transcript cache for the evaluation dataset.

Dev videos already have transcripts in the database (exported read-only by the CLI). Test videos
are fetched from YouTube under the same rules as the local worker: at most a few per day
(``DAILY_FETCH_LIMIT``, which together with the worker stays within 10 caption requests a day),
1.5 s apart, and a block stops everything instead of being worked around.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from yla.eval.dataset import Cache, CachedVideo, Dataset
from yla.youtube.transcripts import TranscriptProvider, TranscriptResult, TranscriptStatus

DAILY_FETCH_LIMIT = 5
REQUEST_INTERVAL_SECONDS = 1.5


@dataclass(frozen=True)
class VideoInfo:
    channel: str
    title: str
    description: str


@dataclass
class FetchReport:
    exported: list[str] = field(default_factory=list)
    fetched: dict[str, str] = field(default_factory=dict)  # video_id -> transcript status
    missing_info: list[str] = field(default_factory=list)
    left_for_later: list[str] = field(default_factory=list)
    blocked: bool = False


def fetched_today(cache: Cache, *, now: datetime, tz: ZoneInfo) -> int:
    today = now.astimezone(tz).date()
    return sum(
        1 for v in cache.all() if v.source == "youtube" and v.fetched_at.astimezone(tz).date() == today
    )


def fill_cache(
    dataset: Dataset,
    cache: Cache,
    *,
    info: dict[str, VideoInfo],
    stored: dict[str, TranscriptResult],
    provider: TranscriptProvider | None,
    now: Callable[[], datetime],
    tz: ZoneInfo,
    daily_limit: int = DAILY_FETCH_LIMIT,
    sleep: Callable[[float], None] = time.sleep,
) -> FetchReport:
    """``stored``: transcripts already in the database; ``provider``: None to export only."""
    report = FetchReport()
    budget = daily_limit - fetched_today(cache, now=now(), tz=tz)
    for item in dataset.items:
        if cache.get(item.video_id) is not None:
            continue
        meta = info.get(item.video_id)
        if meta is None:
            report.missing_info.append(item.video_id)
            continue

        source = "database"
        result = stored.get(item.video_id)
        if result is None:
            if provider is None or budget <= 0 or report.blocked:
                report.left_for_later.append(item.video_id)
                continue
            if report.fetched:
                sleep(REQUEST_INTERVAL_SECONDS)
            result = provider.fetch(item.video_id)
            budget -= 1
            source = "youtube"
            report.fetched[item.video_id] = result.status.value
            if result.status is TranscriptStatus.BLOCKED:
                report.blocked = True
                continue
            if result.status is TranscriptStatus.ERROR:
                continue  # transient: try again another day
        else:
            report.exported.append(item.video_id)

        cache.put(
            CachedVideo(
                video_id=item.video_id,
                channel=meta.channel,
                title=meta.title,
                description=meta.description,
                transcript_status=result.status,
                language=result.language,
                text=result.text or None,
                source=source,
                fetched_at=now(),
            )
        )
    return report
