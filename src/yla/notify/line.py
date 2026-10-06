"""LINE Messaging API push (plain text).

One push carries up to 5 text messages and counts as one message against the free monthly
quota (200), whatever the number of message objects. ``X-Line-Retry-Key`` makes a re-push of
the same digest safe: LINE answers 409 instead of delivering it again.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

PUSH_URL = "https://api.line.me/v2/bot/message/push"
MAX_MESSAGES_PER_PUSH = 5
MAX_TEXT_LENGTH = 5000  # LINE's limit, counted in UTF-16 code units


class NotifyError(Exception):
    pass


class _Transient(NotifyError):
    pass


class Notifier(Protocol):
    def push(self, messages: list[str], *, retry_key: str) -> bool:
        """Deliver the messages. Returns False if this retry key was already delivered."""
        ...


def utf16_length(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


class LineNotifier:
    def __init__(
        self,
        access_token: str,
        to: str,
        client: httpx.Client,
        *,
        attempts: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        self._token = access_token
        self._to = to
        self._client = client
        self._post = retry(
            retry=retry_if_exception_type((_Transient, httpx.TransportError)),
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=backoff_seconds),
            reraise=True,
        )(self._post_once)

    def push(self, messages: list[str], *, retry_key: str) -> bool:
        if not 1 <= len(messages) <= MAX_MESSAGES_PER_PUSH:
            raise ValueError(f"LINE accepts 1-{MAX_MESSAGES_PER_PUSH} messages per push, got {len(messages)}")
        too_long = [i for i, m in enumerate(messages) if utf16_length(m) > MAX_TEXT_LENGTH]
        if too_long:
            raise ValueError(f"messages {too_long} exceed {MAX_TEXT_LENGTH} characters")

        payload: dict[str, Any] = {
            "to": self._to,
            "messages": [{"type": "text", "text": m} for m in messages],
        }
        try:
            return self._post(payload, retry_key)
        except httpx.TransportError as exc:
            raise NotifyError(f"network error: {exc}") from exc

    def _post_once(self, payload: dict[str, Any], retry_key: str) -> bool:
        response = self._client.post(
            PUSH_URL,
            json=payload,
            headers={"Authorization": f"Bearer {self._token}", "X-Line-Retry-Key": retry_key},
        )
        if response.status_code == 200:
            return True
        if response.status_code == 409:  # same retry key already accepted: delivered before
            return False
        detail = response.text[:300]
        if response.status_code >= 500:
            raise _Transient(f"HTTP {response.status_code}: {detail}")
        # 400 bad request, 401 bad token, 429 rate limit or monthly quota used up.
        raise NotifyError(f"HTTP {response.status_code}: {detail}")
