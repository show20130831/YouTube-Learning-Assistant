from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from yla.db.models import LiveStatus
from yla.youtube.data_api import VIDEOS_URL, MetadataError, YouTubeDataApi, parse_duration


@pytest.mark.parametrize(
    ("value", "seconds"),
    [("PT54S", 54), ("PT1M13S", 73), ("PT2H", 7200), ("PT1H2M3S", 3723), ("P1DT2H", 93600), ("P0D", 0)],
)
def test_parse_duration(value: str, seconds: int) -> None:
    assert parse_duration(value) == seconds


def test_parse_duration_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        parse_duration("54 seconds")


def item(video_id: str, duration: str = "PT10M", live: str = "none") -> dict[str, Any]:
    return {
        "id": video_id,
        "contentDetails": {"duration": duration},
        "snippet": {"liveBroadcastContent": live},
    }


@pytest.fixture
def api() -> YouTubeDataApi:
    return YouTubeDataApi("secret-key", httpx.Client(), backoff_seconds=0)


@respx.mock
def test_fetch_sends_key_in_header_not_url(api: YouTubeDataApi) -> None:
    route = respx.get(VIDEOS_URL).respond(
        200, json={"items": [item("a", "PT54S"), item("b", "P0D", "upcoming"), item("c", live="weird")]}
    )
    result = api.fetch(["a", "b", "c", "gone", "a"])

    assert result["a"].duration_seconds == 54
    assert result["b"].live_status is LiveStatus.UPCOMING
    assert result["c"].live_status is LiveStatus.NONE
    assert "gone" not in result
    request = route.calls.last.request
    assert request.headers["X-Goog-Api-Key"] == "secret-key"
    assert "secret-key" not in str(request.url)
    assert request.url.params["id"] == "a,b,c,gone"  # deduplicated


@respx.mock
def test_fetch_batches_fifty_ids_per_call(api: YouTubeDataApi) -> None:
    route = respx.get(VIDEOS_URL).respond(200, json={"items": []})
    api.fetch([f"v{i}" for i in range(120)])
    assert [len(c.request.url.params["id"].split(",")) for c in route.calls] == [50, 50, 20]


@respx.mock
def test_server_errors_are_retried(api: YouTubeDataApi) -> None:
    route = respx.get(VIDEOS_URL)
    route.side_effect = [
        httpx.Response(503),
        httpx.ConnectError("x"),
        httpx.Response(200, json={"items": []}),
    ]
    assert api.fetch(["a"]) == {}
    assert route.call_count == 3


@respx.mock
@pytest.mark.parametrize(
    ("status", "reason"),
    [(403, "quotaExceeded"), (400, "keyInvalid"), (403, "accessNotConfigured")],
)
def test_client_errors_raise_metadata_error_without_retry(
    api: YouTubeDataApi, status: int, reason: str
) -> None:
    route = respx.get(VIDEOS_URL).respond(
        status, json={"error": {"message": "nope", "errors": [{"reason": reason}]}}
    )
    with pytest.raises(MetadataError, match=reason):
        api.fetch(["a"])
    assert route.call_count == 1


@respx.mock
def test_persistent_outage_raises_metadata_error(api: YouTubeDataApi) -> None:
    respx.get(VIDEOS_URL).respond(500)
    with pytest.raises(MetadataError):
        api.fetch(["a"])


@respx.mock
def test_non_json_error_body(api: YouTubeDataApi) -> None:
    respx.get(VIDEOS_URL).respond(403, text="<html>forbidden</html>")
    with pytest.raises(MetadataError, match="HTTP 403"):
        api.fetch(["a"])
