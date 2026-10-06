"""Transcript (caption) providers and sinks.

YouTube blocks caption requests from cloud IPs (Phase 0: 15/15 blocked on Modal), so:
- the local worker uses ``YouTubeTranscriptProvider`` from a home network and writes results
  through a ``TranscriptSink``;
- Modal only reads stored transcripts via ``DatabaseTranscriptProvider`` and never calls YouTube.

``youtube-transcript-api`` is unofficial: requests are rate-limited, never proxied, and a block
is accepted (the video falls back to title + description) rather than worked around.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Protocol

import requests
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session
from youtube_transcript_api import (
    AgeRestricted,
    IpBlocked,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    VideoUnplayable,
    YouTubeRequestFailed,
    YouTubeTranscriptApi,
    YouTubeTranscriptApiException,
)

from yla.db.models import FetchedBy, SourceType, Transcript

_TOO_MANY_REQUESTS = re.compile(r"\b429\b")
_SERVER_ERROR = re.compile(r"\b5\d\d\b")

# English first (most tracked channels), then Traditional / other Chinese.
PREFERRED_LANGUAGES = ("en", "en-US", "en-GB", "zh-TW", "zh-Hant", "zh", "zh-Hans", "zh-CN")


class TranscriptStatus(StrEnum):
    MANUAL = "manual"
    GENERATED = "generated"
    NONE = "none"  # the video has no usable captions
    BLOCKED = "blocked"  # YouTube refused the request (IP block / rate limit)
    ERROR = "error"


class TranscriptResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: TranscriptStatus
    language: str | None = None
    text: str = ""
    error: str | None = None

    @property
    def has_text(self) -> bool:
        return self.status in (TranscriptStatus.MANUAL, TranscriptStatus.GENERATED) and bool(
            self.text.strip()
        )


class TranscriptProvider(Protocol):
    def fetch(self, video_id: str) -> TranscriptResult: ...


class TranscriptSink(Protocol):
    """Where the local worker stores transcripts. v1 writes to the DB directly (self-hosted:
    the worker and the database belong to the same person); an HTTP upload could replace it."""

    def save(self, video_pk: int, result: TranscriptResult) -> None: ...


def _describe(exc: BaseException) -> str:
    if isinstance(exc, YouTubeRequestFailed):
        # e.g. "502 Server Error: Bad Gateway"; drop the long signed URL that follows.
        detail = exc.reason.split(" for url:")[0]
    else:
        lines = str(exc).strip().splitlines()
        detail = lines[0] if lines else ""
    return f"{type(exc).__name__}: {detail[:200]}"


def _classify_error(exc: Exception) -> TranscriptStatus:
    if isinstance(exc, RequestBlocked | IpBlocked):
        return TranscriptStatus.BLOCKED
    if isinstance(exc, YouTubeRequestFailed):
        # The library keeps only the HTTP error's message, e.g. "429 Client Error: Too Many Requests".
        return TranscriptStatus.BLOCKED if _TOO_MANY_REQUESTS.search(exc.reason) else TranscriptStatus.ERROR
    if isinstance(
        exc, TranscriptsDisabled | NoTranscriptFound | VideoUnavailable | VideoUnplayable | AgeRestricted
    ):
        return TranscriptStatus.NONE
    return TranscriptStatus.ERROR


def _is_transient(exc: BaseException) -> bool:
    """YouTube-side hiccups (5xx, timeouts, dropped connections) are worth a retry; blocks are not."""
    if isinstance(exc, YouTubeRequestFailed):
        return bool(_SERVER_ERROR.search(exc.reason))
    return isinstance(exc, requests.Timeout | requests.ConnectionError)


def _language_rank(code: str) -> int:
    return PREFERRED_LANGUAGES.index(code) if code in PREFERRED_LANGUAGES else len(PREFERRED_LANGUAGES)


class _TimeoutSession(requests.Session):
    """youtube-transcript-api sets no timeout, so a stalled request could hang the worker forever."""

    def __init__(self, timeout: float) -> None:
        super().__init__()
        self._timeout = timeout

    def request(self, *args: Any, **kwargs: Any) -> requests.Response:
        kwargs.setdefault("timeout", self._timeout)
        return super().request(*args, **kwargs)


class YouTubeTranscriptProvider:
    """Fetches captions with youtube-transcript-api. Manual captions beat auto-generated ones.

    Transient YouTube errors (5xx, timeouts) are retried a couple of times with a pause;
    blocks and "no captions" are returned immediately.
    """

    def __init__(
        self,
        api: Any | None = None,
        *,
        timeout: float = 30.0,
        attempts: int = 3,
        retry_delay: float = 5.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api = api if api is not None else YouTubeTranscriptApi(http_client=_TimeoutSession(timeout))
        self._attempts = attempts
        self._retry_delay = retry_delay
        self._sleep = sleep

    def fetch(self, video_id: str) -> TranscriptResult:
        attempt = 1
        while True:
            try:
                return self._fetch_once(video_id)
            except (YouTubeTranscriptApiException, requests.RequestException) as exc:
                if attempt < self._attempts and _is_transient(exc):
                    self._sleep(self._retry_delay * attempt)
                    attempt += 1
                    continue
                return TranscriptResult(status=_classify_error(exc), error=_describe(exc))

    def _fetch_once(self, video_id: str) -> TranscriptResult:
        tracks = list(self._api.list(video_id))
        if not tracks:
            return TranscriptResult(status=TranscriptStatus.NONE)
        chosen = min(tracks, key=lambda t: (t.is_generated, _language_rank(t.language_code)))
        # One snippet per line, so cleaning can drop the repeated lines auto-captions produce.
        text = "\n".join(s.text for s in chosen.fetch())
        if not text.strip():
            return TranscriptResult(status=TranscriptStatus.NONE, language=chosen.language_code)
        status = TranscriptStatus.GENERATED if chosen.is_generated else TranscriptStatus.MANUAL
        return TranscriptResult(status=status, language=chosen.language_code, text=text)


_STATUS_TO_SOURCE = {
    TranscriptStatus.MANUAL: SourceType.MANUAL_CAPTION,
    TranscriptStatus.GENERATED: SourceType.AUTO_CAPTION,
}
_SOURCE_TO_STATUS = {v: k for k, v in _STATUS_TO_SOURCE.items()}


class DatabaseTranscriptSink:
    """Stores successful transcripts; failures are only reported (nothing to store)."""

    def __init__(self, session: Session, fetched_by: FetchedBy = FetchedBy.LOCAL) -> None:
        self._session = session
        self._fetched_by = fetched_by

    def save(self, video_pk: int, result: TranscriptResult) -> None:
        if not result.has_text:
            return
        row = self._session.get(Transcript, video_pk) or Transcript(video_id=video_pk)
        row.source_type = _STATUS_TO_SOURCE[result.status]
        row.language = result.language or "und"
        row.text = result.text
        row.char_count = len(result.text)
        row.fetched_by = self._fetched_by
        self._session.add(row)
        self._session.flush()


class DatabaseTranscriptProvider:
    """Modal side: serves transcripts the local worker stored. Never contacts YouTube."""

    def __init__(self, session: Session) -> None:
        self._session = session

    def get(self, video_pk: int) -> TranscriptResult:
        row = self._session.get(Transcript, video_pk)
        if row is None or row.source_type not in _SOURCE_TO_STATUS:
            return TranscriptResult(status=TranscriptStatus.NONE)
        return TranscriptResult(
            status=_SOURCE_TO_STATUS[row.source_type], language=row.language, text=row.text
        )
