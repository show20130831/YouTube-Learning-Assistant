from __future__ import annotations

import logging

import httpx
import pytest
import respx

from yla.ops.heartbeat import MAX_MESSAGE_BYTES, ping

URL = "https://hc-ping.com/1234-abcd"


def test_no_url_sends_nothing() -> None:
    with respx.mock(assert_all_called=False) as router:
        assert ping(None, httpx.Client(), ok=True) is False
        assert ping("", httpx.Client(), ok=True) is False
        assert not router.calls


@respx.mock
@pytest.mark.parametrize(("ok", "path"), [(True, URL), (False, f"{URL}/fail")])
def test_success_and_failure_endpoints(ok: bool, path: str) -> None:
    route = respx.post(path).respond(200)
    assert ping(URL + "/", httpx.Client(), ok=ok, message="details") is True
    assert route.calls.last.request.content == b"details"


@respx.mock
def test_message_is_truncated() -> None:
    route = respx.post(URL).respond(200)
    ping(URL, httpx.Client(), ok=True, message="x" * 20_000)
    assert len(route.calls.last.request.content) == MAX_MESSAGE_BYTES


@respx.mock
@pytest.mark.parametrize("failure", [httpx.Response(500), httpx.ConnectError("down")])
def test_ping_failure_is_logged_without_the_url(
    failure: httpx.Response | Exception, caplog: pytest.LogCaptureFixture
) -> None:
    respx.post(URL).mock(side_effect=[failure])
    with caplog.at_level(logging.WARNING):
        assert ping(URL, httpx.Client(), ok=True) is False
    assert "heartbeat ping failed" in caplog.text
    assert URL not in caplog.text
