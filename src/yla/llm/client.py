"""LLM access behind a small protocol, so analysis code never depends on one provider or model.

``OpenRouterClient`` talks to OpenRouter's OpenAI-compatible API with plain httpx and asks for
JSON-schema-constrained output. Callers still validate the result: free models do not always
honour the schema.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol, TypedDict

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$")


class ChatMessage(TypedDict):
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    data: dict[str, Any]
    model: str  # the model that actually answered
    usage: dict[str, Any] = field(default_factory=dict)


class LLMError(Exception):
    """The model could not produce an answer (HTTP error, provider error, empty reply)."""


class LLMRateLimited(LLMError):
    """429: daily free quota used up or the upstream provider is busy. Try another model."""


class LLMOutputInvalid(LLMError):
    """The reply was not parseable JSON."""


class _TransientError(LLMError):
    pass


class LLMClient(Protocol):
    def complete_json(
        self, messages: list[ChatMessage], *, schema: dict[str, Any], schema_name: str, model: str
    ) -> LLMResponse: ...


def parse_json_reply(content: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating the Markdown code fences some models add anyway."""
    text = _CODE_FENCE.sub("", content.strip())
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMOutputInvalid(f"reply is not JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise LLMOutputInvalid("reply is not a JSON object")
    return data


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        client: httpx.Client,
        *,
        temperature: float = 0.2,
        attempts: int = 3,
        backoff_seconds: float = 2.0,
    ) -> None:
        self._api_key = api_key
        self._client = client
        self._temperature = temperature
        self._post = retry(
            retry=retry_if_exception_type((_TransientError, httpx.TransportError)),
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(multiplier=backoff_seconds),
            reraise=True,
        )(self._post_once)

    def complete_json(
        self, messages: list[ChatMessage], *, schema: dict[str, Any], schema_name: str, model: str
    ) -> LLMResponse:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": self._temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
        }
        try:
            body = self._post(payload)
        except httpx.TransportError as exc:
            raise LLMError(f"network error: {exc}") from exc

        choices = body.get("choices") or []
        content = choices[0].get("message", {}).get("content") if choices else None
        if not content:
            raise LLMError("empty reply")
        return LLMResponse(
            data=parse_json_reply(content), model=str(body.get("model", model)), usage=body.get("usage") or {}
        )

    def _post_once(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = self._client.post(
            OPENROUTER_URL,
            json=payload,
            headers={"Authorization": f"Bearer {self._api_key}", "X-Title": "YouTube Learning Assistant"},
        )
        body: dict[str, Any] = response.json() if response.content else {}
        # OpenRouter can also report provider errors inside a 200 response.
        error = body.get("error") or {}
        status = int(error.get("code") or response.status_code) if error else response.status_code
        message = str(error.get("message") or response.reason_phrase)[:300]

        if status == 429:
            raise LLMRateLimited(message)
        if status >= 500:
            raise _TransientError(f"HTTP {status}: {message}")
        if status >= 400:
            raise LLMError(f"HTTP {status}: {message}")
        return body
