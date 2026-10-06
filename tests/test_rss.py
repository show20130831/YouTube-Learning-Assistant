from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
import respx

from yla.youtube.rss import RSS_URL, FeedNotFoundError, FeedParseError, HttpFeedFetcher, parse_feed

FEED_XML = (Path(__file__).parent / "fixtures" / "rss" / "channel_feed.xml").read_bytes()
CHANNEL_ID = "UCexampleChannel0000000a"


def test_parse_feed_extracts_entries() -> None:
    feed = parse_feed(FEED_XML)

    assert feed.channel_title == "Example AI"
    assert [e.video_id for e in feed.entries] == ["vid00000003", "vid00000002", "vid00000001"]
    first = feed.entries[0]
    assert first.channel_id == CHANNEL_ID  # entry-level id keeps the "UC" prefix
    assert first.title == "Building Reliable AI Agents"
    assert first.published_at == datetime(2026, 10, 5, 17, 37, 6, tzinfo=UTC)
    assert "Chapters:" in first.description
    assert first.url == "https://www.youtube.com/watch?v=vid00000003"


def test_empty_description_becomes_empty_string() -> None:
    assert parse_feed(FEED_XML).entries[1].description == ""


def test_entry_without_video_id_is_skipped() -> None:
    xml = FEED_XML.replace(b"<yt:videoId>vid00000002</yt:videoId>", b"")
    assert [e.video_id for e in parse_feed(xml).entries] == ["vid00000003", "vid00000001"]


@pytest.mark.parametrize("payload", [b"not xml", b"<html><body>consent</body></html>"])
def test_non_feed_payload_raises(payload: bytes) -> None:
    with pytest.raises(FeedParseError):
        parse_feed(payload)


def test_entity_expansion_is_refused() -> None:
    bomb = b'<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><feed>&a;</feed>'
    with pytest.raises(FeedParseError):
        parse_feed(bomb)


@pytest.fixture
def fetcher() -> HttpFeedFetcher:
    return HttpFeedFetcher(httpx.Client(), attempts=3, backoff_seconds=0)


@respx.mock
def test_fetch_requests_channel_feed(fetcher: HttpFeedFetcher) -> None:
    route = respx.get(RSS_URL, params={"channel_id": CHANNEL_ID}).respond(200, content=FEED_XML)
    assert len(fetcher.fetch(CHANNEL_ID).entries) == 3
    assert route.call_count == 1


@respx.mock
def test_fetch_404_raises_without_retry(fetcher: HttpFeedFetcher) -> None:
    route = respx.get(RSS_URL).respond(404)
    with pytest.raises(FeedNotFoundError):
        fetcher.fetch(CHANNEL_ID)
    assert route.call_count == 1


@respx.mock
def test_fetch_retries_transient_errors(fetcher: HttpFeedFetcher) -> None:
    route = respx.get(RSS_URL)
    route.side_effect = [
        httpx.ConnectError("boom"),
        httpx.Response(503),
        httpx.Response(200, content=FEED_XML),
    ]
    assert len(fetcher.fetch(CHANNEL_ID).entries) == 3
    assert route.call_count == 3


@respx.mock
def test_fetch_gives_up_after_max_attempts(fetcher: HttpFeedFetcher) -> None:
    route = respx.get(RSS_URL).respond(500)
    with pytest.raises(httpx.HTTPStatusError):
        fetcher.fetch(CHANNEL_ID)
    assert route.call_count == 3
