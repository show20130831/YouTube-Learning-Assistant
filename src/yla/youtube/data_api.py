"""YouTube Data API v3: video length and live status (optional; needs YOUTUBE_API_KEY).

``videos.list`` costs 1 quota unit per call of up to 50 IDs (free quota: 10,000 units/day),
so a day's new videos cost about 1 unit. The key is sent in the ``X-Goog-Api-Key`` header
rather than the ``key`` query parameter, so it never appears in logged request URLs.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol

import httpx
from pydantic import BaseModel, ConfigDict
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from yla.db.models import LiveStatus

VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
MAX_IDS_PER_CALL = 50
_FIELDS = "items(id,contentDetails/duration,snippet/liveBroadcastContent)"
# ISO 8601 durations as YouTube returns them: PT54S, PT1H2M3S, P1DT2H, P0D (upcoming streams).
_LIVE_VALUES = frozenset(s.value for s in LiveStatus)
_DURATION = re.compile(r"^P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?)?$")


class VideoMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    duration_seconds: int
    live_status: LiveStatus


class MetadataError(Exception):
    """The Data API could not be used (bad key, quota exceeded, outage)."""


class MetadataProvider(Protocol):
    def fetch(self, video_ids: Sequence[str]) -> dict[str, VideoMetadata]:
        """Metadata for the IDs YouTube returned; private or deleted videos are simply absent."""
        ...


def parse_duration(value: str) -> int:
    match = _DURATION.match(value)
    if not match:
        raise ValueError(f"unrecognised duration {value!r}")
    days, hours, minutes, seconds = (int(part) if part else 0 for part in match.groups())
    return ((days * 24 + hours) * 60 + minutes) * 60 + seconds


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


def _error_reason(response: httpx.Response) -> str:
    try:
        error: dict[str, Any] = response.json().get("error", {})
    except ValueError:
        return f"HTTP {response.status_code}"
    reasons = [e.get("reason") for e in error.get("errors", []) if e.get("reason")]
    detail = ", ".join(reasons) or str(error.get("message", ""))[:200]
    return f"HTTP {response.status_code}: {detail}"


class YouTubeDataApi:
    def __init__(
        self, api_key: str, client: httpx.Client, *, attempts: int = 3, backoff_seconds: float = 2.0
    ) -> None:
        self._api_key = api_key
        self._client = client
        self._get = retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=backoff_seconds),
            reraise=True,
        )(self._get_once)

    def fetch(self, video_ids: Sequence[str]) -> dict[str, VideoMetadata]:
        ids = list(dict.fromkeys(video_ids))
        result: dict[str, VideoMetadata] = {}
        for start in range(0, len(ids), MAX_IDS_PER_CALL):
            try:
                body = self._get(ids[start : start + MAX_IDS_PER_CALL])
            except (httpx.HTTPError, ValueError) as exc:
                raise MetadataError(str(exc)) from exc
            for item in body.get("items", []):
                live = item.get("snippet", {}).get("liveBroadcastContent", "none")
                result[item["id"]] = VideoMetadata(
                    video_id=item["id"],
                    duration_seconds=parse_duration(item.get("contentDetails", {}).get("duration", "P0D")),
                    live_status=LiveStatus(live) if live in _LIVE_VALUES else LiveStatus.NONE,
                )
        return result

    def _get_once(self, ids: list[str]) -> dict[str, Any]:
        response = self._client.get(
            VIDEOS_URL,
            params={"part": "contentDetails,snippet", "id": ",".join(ids), "fields": _FIELDS},
            headers={"X-Goog-Api-Key": self._api_key},
        )
        if response.status_code >= 500:
            response.raise_for_status()
        if response.status_code != 200:
            # 400 bad key, 403 quota exceeded / API not enabled: retrying will not help.
            raise MetadataError(_error_reason(response))
        body: dict[str, Any] = response.json()
        return body
