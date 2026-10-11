"""Outside check that the daily digest went out (a "dead man's switch").

The system cannot report that it never ran, so an external service expects a ping every day:
Healthchecks.io by default, or anything with the same convention (Better Stack heartbeats,
a self-hosted Healthchecks). Success pings ``HEARTBEAT_URL``; failure pings ``HEARTBEAT_URL/fail``
so the alert is immediate. Without a configured URL nothing is sent, and a failed ping is only
logged: monitoring must never break the job it watches.
"""

from __future__ import annotations

import logging

import httpx

logger = logging.getLogger(__name__)

MAX_MESSAGE_BYTES = 10_000  # Healthchecks stores the first 10 KB of the request body


def ping(url: str | None, client: httpx.Client, *, ok: bool, message: str = "") -> bool:
    if not url:
        return False
    target = url.rstrip("/") + ("" if ok else "/fail")
    try:
        response = client.post(target, content=message.encode("utf-8")[:MAX_MESSAGE_BYTES], timeout=10)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        # The URL identifies the check, so it is not logged.
        logger.warning("heartbeat ping failed: %s", type(exc).__name__)
        return False
    return True
