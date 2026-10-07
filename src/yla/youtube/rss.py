"""YouTube channel RSS (Atom) feeds: the free, quota-less way to detect new videos.

A feed lists a channel's latest 15 uploads with title, publish time and description.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Protocol
from xml.etree.ElementTree import Element, ParseError

import httpx
from defusedxml import DefusedXmlException
from defusedxml.ElementTree import fromstring
from pydantic import AwareDatetime, BaseModel, ConfigDict
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

RSS_URL = "https://www.youtube.com/feeds/videos.xml"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "yt": "http://www.youtube.com/xml/schemas/2015",
    "media": "http://search.yahoo.com/mrss/",
}


class FeedEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    video_id: str
    channel_id: str
    title: str
    description: str
    published_at: AwareDatetime

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


class Feed(BaseModel):
    model_config = ConfigDict(frozen=True)

    channel_title: str | None
    entries: list[FeedEntry]


class FeedParseError(ValueError):
    """The response was not a usable YouTube feed."""


class FeedNotFoundError(LookupError):
    """YouTube kept returning 404: the channel ID is probably wrong or the channel is gone."""


class FeedFetcher(Protocol):
    def fetch(self, channel_id: str) -> Feed: ...


def _text(element: Element, path: str) -> str | None:
    found = element.find(path, NS)
    return found.text if found is not None and found.text is not None else None


def parse_feed(xml: str | bytes) -> Feed:
    try:
        root = fromstring(xml)
    except (ParseError, DefusedXmlException) as exc:  # malformed, or DTD/entity tricks
        raise FeedParseError(f"invalid XML: {exc}") from exc
    if root.tag != f"{{{NS['atom']}}}feed":
        raise FeedParseError(f"unexpected root element {root.tag!r}")

    entries: list[FeedEntry] = []
    for node in root.findall("atom:entry", NS):
        video_id = _text(node, "yt:videoId")
        # Use the entry's channel ID: the feed-level one omits the "UC" prefix.
        channel_id = _text(node, "yt:channelId")
        published = _text(node, "atom:published")
        if not (video_id and channel_id and published):
            logger.warning("skipping feed entry without id/channel/published")
            continue
        entries.append(
            FeedEntry(
                video_id=video_id,
                channel_id=channel_id,
                title=_text(node, "atom:title") or "",
                description=_text(node, "media:group/media:description") or "",
                published_at=datetime.fromisoformat(published),
            )
        )
    return Feed(channel_title=_text(root, "atom:title"), entries=entries)


def _is_retryable(exc: BaseException) -> bool:
    # The feed endpoint also returns spurious 404s (observed 2026-10-07: the same valid channel
    # alternated 404/200 within seconds), so a 404 is retried and only reported if it persists.
    if isinstance(exc, FeedNotFoundError):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500 or exc.response.status_code == 429
    return isinstance(exc, httpx.TransportError)


class HttpFeedFetcher:
    """Fetches feeds over HTTP, retrying transient failures (network errors, 404, 429, 5xx)."""

    def __init__(
        self,
        client: httpx.Client,
        *,
        attempts: int = 8,
        backoff_seconds: float = 2.0,
        max_wait_seconds: float = 10.0,
    ) -> None:
        # During YouTube feed outages only ~1 in 4 requests succeeds (measured 2026-10-07), so
        # several short-capped retries are needed; worst case is about a minute per channel.
        self._client = client
        self._fetch = retry(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=backoff_seconds, max=max_wait_seconds),
            reraise=True,
        )(self._fetch_once)

    def fetch(self, channel_id: str) -> Feed:
        return self._fetch(channel_id)

    def _fetch_once(self, channel_id: str) -> Feed:
        response = self._client.get(RSS_URL, params={"channel_id": channel_id})
        if response.status_code == 404:
            raise FeedNotFoundError(f"no feed for channel {channel_id}")
        response.raise_for_status()
        return parse_feed(response.content)
