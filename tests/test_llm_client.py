import json
from typing import Any

import httpx
import pytest
import respx

from yla.llm.client import (
    OPENROUTER_URL,
    LLMError,
    LLMOutputInvalid,
    LLMRateLimited,
    OpenRouterClient,
    parse_json_reply,
)

SCHEMA: dict[str, Any] = {"type": "object", "properties": {"a": {"type": "integer"}}}


def reply(content: str, model: str = "m:free") -> dict[str, Any]:
    return {"model": model, "choices": [{"message": {"content": content}}], "usage": {"total_tokens": 9}}


@pytest.fixture
def client() -> OpenRouterClient:
    return OpenRouterClient("sk-test", httpx.Client(), backoff_seconds=0)


def call(client: OpenRouterClient) -> Any:
    return client.complete_json(
        [{"role": "user", "content": "hi"}], schema=SCHEMA, schema_name="t", model="m:free"
    )


@respx.mock
def test_sends_schema_and_parses_reply(client: OpenRouterClient) -> None:
    route = respx.post(OPENROUTER_URL).respond(200, json=reply('{"a": 1}', model="m:free"))
    response = call(client)

    assert (response.data, response.model, response.usage) == ({"a": 1}, "m:free", {"total_tokens": 9})
    sent = json.loads(route.calls.last.request.content)
    assert sent["model"] == "m:free"
    assert sent["response_format"]["json_schema"] == {"name": "t", "strict": True, "schema": SCHEMA}
    assert route.calls.last.request.headers["Authorization"] == "Bearer sk-test"


@respx.mock
def test_rate_limit_is_not_retried(client: OpenRouterClient) -> None:
    route = respx.post(OPENROUTER_URL).respond(429, json={"error": {"code": 429, "message": "slow down"}})
    with pytest.raises(LLMRateLimited, match="slow down"):
        call(client)
    assert route.call_count == 1


@respx.mock
def test_provider_error_inside_200_is_detected(client: OpenRouterClient) -> None:
    respx.post(OPENROUTER_URL).respond(200, json={"error": {"code": 429, "message": "upstream busy"}})
    with pytest.raises(LLMRateLimited):
        call(client)


@respx.mock
def test_server_errors_are_retried(client: OpenRouterClient) -> None:
    route = respx.post(OPENROUTER_URL)
    route.side_effect = [
        httpx.Response(502, json={}),
        httpx.ConnectError("x"),
        httpx.Response(200, json=reply("{}")),
    ]
    assert call(client).data == {}
    assert route.call_count == 3


@respx.mock
def test_persistent_network_error_becomes_llm_error(client: OpenRouterClient) -> None:
    respx.post(OPENROUTER_URL).side_effect = httpx.ConnectError("offline")
    with pytest.raises(LLMError, match="network error"):
        call(client)


@respx.mock
@pytest.mark.parametrize(
    ("response", "error"),
    [
        (httpx.Response(400, json={"error": {"code": 400, "message": "bad schema"}}), LLMError),
        (httpx.Response(200, json={"choices": []}), LLMError),
        (httpx.Response(200, json=reply("")), LLMError),
        (httpx.Response(200, json=reply("not json")), LLMOutputInvalid),
    ],
)
def test_unusable_replies_raise(
    client: OpenRouterClient, response: httpx.Response, error: type[Exception]
) -> None:
    respx.post(OPENROUTER_URL).mock(return_value=response)
    with pytest.raises(error):
        call(client)


@pytest.mark.parametrize("content", ['{"a": 1}', '```json\n{"a": 1}\n```', '  ```\n{"a": 1}```  '])
def test_parse_json_reply_tolerates_code_fences(content: str) -> None:
    assert parse_json_reply(content) == {"a": 1}


def test_parse_json_reply_rejects_non_objects() -> None:
    with pytest.raises(LLMOutputInvalid):
        parse_json_reply("[1, 2]")
