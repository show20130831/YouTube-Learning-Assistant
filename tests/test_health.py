from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from pydantic import SecretStr
from sqlalchemy import Engine
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from yla import cli
from yla.config import AppSettings, Secrets
from yla.db.models import JobRun, JobStatus
from yla.ops.health import (
    LINE_API,
    OPENROUTER_API,
    YOUTUBE_CHANNELS_URL,
    CheckResult,
    Status,
    check_database_size,
    check_heartbeat,
    check_job_runs,
    check_line,
    check_openrouter,
    check_youtube_api,
    run_health_checks,
)
from yla.ops.jobs import config_fingerprint

NOW = datetime.now(UTC)
MODEL = "vendor/model:free"


def by_name(results: list[CheckResult]) -> dict[str, CheckResult]:
    return {r.name: r for r in results}


def model_entry(model_id: str = MODEL, **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "id": model_id,
        "pricing": {"prompt": "0", "completion": "0"},
        "supported_parameters": ["response_format"],
        "context_length": 262144,
    }
    entry.update(overrides)
    return entry


# --- LINE -----------------------------------------------------------------------------------


@respx.mock
def test_line_ok_with_usage() -> None:
    respx.get(f"{LINE_API}/info").respond(200, json={"displayName": "Bot"})
    respx.get(f"{LINE_API}/message/quota").respond(200, json={"type": "limited", "value": 200})
    respx.get(f"{LINE_API}/message/quota/consumption").respond(200, json={"totalUsage": 170})
    results = by_name(check_line("token", httpx.Client()))
    assert results["LINE"].status is Status.OK
    usage = results["LINE messages this month"]
    assert (usage.status, usage.detail) == (Status.WARN, "170/200")


@respx.mock
def test_line_rejected_or_missing_token() -> None:
    respx.get(f"{LINE_API}/info").respond(401)
    assert check_line("bad", httpx.Client())[0].status is Status.FAIL
    assert check_line(None, httpx.Client())[0].status is Status.FAIL


# --- OpenRouter ---------------------------------------------------------------------------


@respx.mock
def test_openrouter_models_checked_without_calling_them() -> None:
    respx.get(f"{OPENROUTER_API}/key").respond(200, json={"data": {"is_free_tier": True, "usage": 0}})
    chat = respx.post(f"{OPENROUTER_API}/chat/completions")
    respx.get(f"{OPENROUTER_API}/models").respond(
        200,
        json={
            "data": [
                model_entry(),
                model_entry("paid/model", pricing={"prompt": "0.000001", "completion": "0"}),
                model_entry("plain/model", supported_parameters=["temperature"]),
            ]
        },
    )
    models = [MODEL, "paid/model", "plain/model", "gone/model"]
    results = by_name(check_openrouter("sk", models, httpx.Client()))

    assert results["OpenRouter key"].status is Status.OK
    assert results[f"Model {MODEL}"].status is Status.OK
    assert results["Model paid/model"].detail == "no longer free"
    assert results["Model plain/model"].detail == "no structured output support"
    assert results["Model gone/model"].status is Status.FAIL
    assert not chat.called  # health checks never spend LLM quota


@respx.mock
def test_openrouter_key_problems() -> None:
    respx.get(f"{OPENROUTER_API}/key").respond(401)
    respx.get(f"{OPENROUTER_API}/models").respond(200, json={"data": [model_entry()]})
    assert by_name(check_openrouter("bad", [MODEL], httpx.Client()))["OpenRouter key"].status is Status.FAIL
    assert by_name(check_openrouter(None, [MODEL], httpx.Client()))["OpenRouter key"].status is Status.FAIL
    assert by_name(check_openrouter("sk", [], httpx.Client()))["LLM models"].status is Status.FAIL


# --- YouTube, heartbeat --------------------------------------------------------------------


@respx.mock
def test_youtube_api() -> None:
    assert check_youtube_api(None, None, httpx.Client())[0].status is Status.INFO
    route = respx.get(YOUTUBE_CHANNELS_URL).mock(
        side_effect=[httpx.Response(200, json={}), httpx.Response(400)]
    )
    assert check_youtube_api("key", "UCx", httpx.Client())[0].status is Status.OK
    assert route.calls.last.request.headers["X-Goog-Api-Key"] == "key"
    assert check_youtube_api("bad", "UCx", httpx.Client())[0].status is Status.FAIL


def test_heartbeat() -> None:
    assert check_heartbeat("https://hc-ping.com/x")[0].status is Status.OK
    assert check_heartbeat(None)[0].status is Status.WARN


# --- job runs and drift ---------------------------------------------------------------------


def settings() -> AppSettings:
    return AppSettings.model_validate(
        {"channels": [{"handle": "LangChain"}], "llm": {"primary_model": MODEL}}
    )


@pytest.mark.db
def test_job_runs_healthy_and_in_sync(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YLA_GIT_SHA", "abc1234")
    cloud_stats = {"config_fingerprint": config_fingerprint(settings()), "git_sha": "abc1234"}
    for name in ("local_worker", "daily_pipeline", "send_digest"):
        stats = cloud_stats if name == "daily_pipeline" else {}
        session.add(JobRun(job_name=name, status=JobStatus.SUCCEEDED, stats=stats))
    session.flush()

    results = by_name(check_job_runs(session, settings(), now=NOW))
    assert results["Stuck runs"].status is Status.OK
    for name in ("local_worker", "daily_pipeline", "send_digest"):
        assert results[f"Last {name}"].status is Status.OK
    assert results["Cloud settings"].status is Status.OK
    assert results["Cloud code version"].detail == "abc1234"


@pytest.mark.db
def test_job_runs_problems(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YLA_GIT_SHA", "new5678")
    session.add_all(
        [
            JobRun(
                job_name="send_digest",
                status=JobStatus.RUNNING,
                stats={},
                started_at=NOW - timedelta(hours=1),
            ),
            JobRun(
                job_name="daily_pipeline",
                status=JobStatus.FAILED,
                error="InternalError: boom",
                stats={"config_fingerprint": "old", "git_sha": "old1234"},
            ),
        ]
    )
    session.flush()

    results = by_name(check_job_runs(session, settings(), now=NOW))
    assert results["Stuck runs"].status is Status.WARN
    assert results["Last local_worker"].detail == "no run in the last 26 hours"
    assert results["Last daily_pipeline"].detail == "failed: InternalError: boom"
    assert results["Cloud settings"].status is Status.WARN
    assert "modal deploy" in results["Cloud settings"].detail
    assert results["Cloud code version"].detail == "cloud old1234, local new5678"


class FakeSession:
    def __init__(self, size: int) -> None:
        self.size = size

    def scalar(self, _: Any) -> int:
        return self.size


@pytest.mark.parametrize(("size", "status"), [(9 * 1024**2, Status.INFO), (300 * 1024**2, Status.WARN)])
def test_database_size(size: int, status: Status) -> None:
    [result] = check_database_size(FakeSession(size))  # type: ignore[arg-type]
    assert result.status is status


# --- orchestration and CLI ----------------------------------------------------------------


@respx.mock
def test_one_broken_check_does_not_hide_the_others() -> None:
    respx.get(f"{LINE_API}/info").mock(side_effect=httpx.ConnectError("down"))
    respx.get(f"{OPENROUTER_API}/key").respond(200, json={"data": {}})
    respx.get(f"{OPENROUTER_API}/models").respond(200, json={"data": [model_entry()]})
    secrets = Secrets(
        _env_file=None,
        line_channel_access_token=SecretStr("t"),
        openrouter_api_key=SecretStr("sk"),
    )
    results = by_name(
        run_health_checks(secrets=secrets, settings=settings(), session=None, client=httpx.Client(), now=NOW)
    )
    assert results["LINE"].status is Status.FAIL
    assert "ConnectError" in results["LINE"].detail
    assert results["OpenRouter key"].status is Status.OK
    assert results["Database"].detail == "DATABASE_URL is not set"


@pytest.mark.db
@respx.mock
def test_cli_health_exit_codes(
    engine: Engine, test_database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("LINE_CHANNEL_ACCESS_TOKEN", "OPENROUTER_API_KEY", "YOUTUBE_API_KEY", "HEARTBEAT_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    respx.get(f"{OPENROUTER_API}/models").respond(200, json={"data": []})
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(
        f"channels: [{{handle: x}}]\nllm: {{primary_model: '{MODEL}'}}\n", encoding="utf-8"
    )

    result = CliRunner().invoke(cli.app, ["health", "--path", str(settings_file)])
    assert result.exit_code == 1  # LINE and OpenRouter keys missing, model not listed
    assert "FAIL  LINE: LINE_CHANNEL_ACCESS_TOKEN is not set" in result.output
    assert "OK    Database: connected" in result.output
    assert "Usage" in result.output
