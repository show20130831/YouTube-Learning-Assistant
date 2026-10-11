"""`yla health`: connectivity, configuration and usage checks, run on demand.

Never calls an LLM (that would spend the free quota); whether models actually answer is
measured by the daily runs (see job stats ``model_outcomes``). Each check is isolated, so one
failing service does not hide the others. Usage that has no reliable API (Modal credits, the
YouTube Data API quota) is listed as "check the console" instead of being guessed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any

import httpx
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from yla.config import AppSettings, Secrets
from yla.db.models import JobRun, JobStatus
from yla.ops.jobs import GRACE, JOB_TIME_LIMITS, config_fingerprint, git_sha

LINE_API = "https://api.line.me/v2/bot"
OPENROUTER_API = "https://openrouter.ai/api/v1"
YOUTUBE_CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"

LINE_MONTHLY_FREE = 200
LINE_USAGE_WARN = 160
DB_FREE_BYTES = 512 * 1024**2  # Neon free tier storage
DB_WARN_BYTES = 250 * 1024**2
RECENT_RUN_WINDOW = timedelta(hours=26)


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    INFO = "info"


class Section(StrEnum):
    CONNECTIVITY = "Connectivity and configuration"
    USAGE = "Usage"


@dataclass(frozen=True)
class CheckResult:
    section: Section
    name: str
    status: Status
    detail: str


def _secret(value: Any) -> str | None:
    raw = value.get_secret_value() if value is not None else None
    return raw or None


# --- connectivity and configuration ----------------------------------------------------------


def check_database(session: Session) -> list[CheckResult]:
    session.execute(text("SELECT 1"))
    return [CheckResult(Section.CONNECTIVITY, "Database", Status.OK, "connected")]


def check_line(token: str | None, client: httpx.Client) -> list[CheckResult]:
    if not token:
        return [
            CheckResult(Section.CONNECTIVITY, "LINE", Status.FAIL, "LINE_CHANNEL_ACCESS_TOKEN is not set")
        ]
    headers = {"Authorization": f"Bearer {token}"}
    info = client.get(f"{LINE_API}/info", headers=headers)
    if info.status_code != 200:
        return [
            CheckResult(
                Section.CONNECTIVITY, "LINE", Status.FAIL, f"token rejected (HTTP {info.status_code})"
            )
        ]
    results = [
        CheckResult(Section.CONNECTIVITY, "LINE", Status.OK, f"bot {info.json().get('displayName', '?')}")
    ]

    quota = client.get(f"{LINE_API}/message/quota", headers=headers).json()
    used = client.get(f"{LINE_API}/message/quota/consumption", headers=headers).json().get("totalUsage", 0)
    limit = quota.get("value", LINE_MONTHLY_FREE) if quota.get("type") == "limited" else None
    status = Status.WARN if limit and used > LINE_USAGE_WARN * limit / LINE_MONTHLY_FREE else Status.INFO
    results.append(
        CheckResult(Section.USAGE, "LINE messages this month", status, f"{used}/{limit or 'unlimited'}")
    )
    return results


def check_openrouter(key: str | None, models: list[str], client: httpx.Client) -> list[CheckResult]:
    results: list[CheckResult] = []
    if not key:
        results.append(
            CheckResult(Section.CONNECTIVITY, "OpenRouter key", Status.FAIL, "OPENROUTER_API_KEY is not set")
        )
    else:
        response = client.get(f"{OPENROUTER_API}/key", headers={"Authorization": f"Bearer {key}"})
        if response.status_code != 200:
            results.append(
                CheckResult(
                    Section.CONNECTIVITY,
                    "OpenRouter key",
                    Status.FAIL,
                    f"rejected (HTTP {response.status_code})",
                )
            )
        else:
            data = response.json().get("data", {})
            results.append(CheckResult(Section.CONNECTIVITY, "OpenRouter key", Status.OK, "valid"))
            tier = "free tier" if data.get("is_free_tier") else "paid tier"
            results.append(
                CheckResult(Section.USAGE, "OpenRouter", Status.INFO, f"{tier}, usage {data.get('usage', 0)}")
            )

    if not models:
        results.append(
            CheckResult(Section.CONNECTIVITY, "LLM models", Status.FAIL, "llm.primary_model is not set")
        )
        return results
    catalogue = {m["id"]: m for m in client.get(f"{OPENROUTER_API}/models").json().get("data", [])}
    for model in models:
        entry = catalogue.get(model)
        if entry is None:
            results.append(
                CheckResult(
                    Section.CONNECTIVITY, f"Model {model}", Status.FAIL, "no longer listed by OpenRouter"
                )
            )
            continue
        pricing = entry.get("pricing", {})
        params = entry.get("supported_parameters") or []
        problems = []
        if str(pricing.get("prompt", "0")) != "0" or str(pricing.get("completion", "0")) != "0":
            problems.append("no longer free")
        if "response_format" not in params and "structured_outputs" not in params:
            problems.append("no structured output support")
        status = Status.WARN if problems else Status.OK
        detail = ", ".join(problems) or f"listed, free, {entry.get('context_length', '?')} tokens context"
        results.append(CheckResult(Section.CONNECTIVITY, f"Model {model}", status, detail))
    return results


def check_youtube_api(key: str | None, channel_id: str | None, client: httpx.Client) -> list[CheckResult]:
    if not key:
        return [
            CheckResult(Section.CONNECTIVITY, "YouTube Data API", Status.INFO, "not configured (optional)")
        ]
    response = client.get(
        YOUTUBE_CHANNELS_URL,
        params={"part": "id", "id": channel_id or "UC_x5XG1OV2P6uZZ5FSM9Ttw"},
        headers={"X-Goog-Api-Key": key},
    )
    if response.status_code != 200:
        return [
            CheckResult(
                Section.CONNECTIVITY,
                "YouTube Data API",
                Status.FAIL,
                f"key rejected (HTTP {response.status_code})",
            )
        ]
    return [CheckResult(Section.CONNECTIVITY, "YouTube Data API", Status.OK, "key valid (used 1 quota unit)")]


def check_heartbeat(url: str | None) -> list[CheckResult]:
    if url:
        return [CheckResult(Section.CONNECTIVITY, "Heartbeat", Status.OK, "configured")]
    return [
        CheckResult(
            Section.CONNECTIVITY,
            "Heartbeat",
            Status.WARN,
            "HEARTBEAT_URL not set: a missing digest goes unnoticed",
        )
    ]


def check_job_runs(session: Session, settings: AppSettings | None, *, now: datetime) -> list[CheckResult]:
    results = []
    stuck = 0
    for name, limit in JOB_TIME_LIMITS.items():
        stuck += (
            session.scalar(
                select(func.count(JobRun.id)).where(
                    JobRun.job_name == name,
                    JobRun.status == JobStatus.RUNNING,
                    JobRun.started_at < now - limit - GRACE,
                )
            )
            or 0
        )
    results.append(
        CheckResult(
            Section.CONNECTIVITY,
            "Stuck runs",
            Status.WARN if stuck else Status.OK,
            f"{stuck} run(s) left 'running' (cleared when that job next starts)" if stuck else "none",
        )
    )

    latest: dict[str, JobRun | None] = {}
    for name in ("local_worker", "daily_pipeline", "send_digest"):
        run = session.scalars(
            select(JobRun).where(JobRun.job_name == name).order_by(JobRun.id.desc()).limit(1)
        ).first()
        latest[name] = run
        if run is None or run.started_at < now - RECENT_RUN_WINDOW:
            results.append(
                CheckResult(Section.CONNECTIVITY, f"Last {name}", Status.WARN, "no run in the last 26 hours")
            )
        elif run.status is JobStatus.FAILED:
            results.append(
                CheckResult(
                    Section.CONNECTIVITY, f"Last {name}", Status.WARN, f"failed: {(run.error or '')[:80]}"
                )
            )
        else:
            results.append(
                CheckResult(
                    Section.CONNECTIVITY,
                    f"Last {name}",
                    Status.OK,
                    f"{run.status.value} at {run.started_at:%m/%d %H:%M} UTC",
                )
            )

    cloud = latest["daily_pipeline"]
    deployed_fp = cloud.stats.get("config_fingerprint") if cloud else None
    if settings is not None and deployed_fp:
        same = deployed_fp == config_fingerprint(settings)
        results.append(
            CheckResult(
                Section.CONNECTIVITY,
                "Cloud settings",
                Status.OK if same else Status.WARN,
                "match settings.yaml"
                if same
                else "differ from settings.yaml: run `uv run modal deploy src/yla/modal_app.py`",
            )
        )
    else:
        results.append(
            CheckResult(Section.CONNECTIVITY, "Cloud settings", Status.INFO, "no recorded cloud run yet")
        )

    deployed_sha = cloud.stats.get("git_sha") if cloud else None
    local_sha = git_sha()
    if deployed_sha:
        same = deployed_sha == local_sha
        results.append(
            CheckResult(
                Section.CONNECTIVITY,
                "Cloud code version",
                Status.OK if same else Status.INFO,
                f"{deployed_sha}" if same else f"cloud {deployed_sha}, local {local_sha}",
            )
        )
    return results


def check_database_size(session: Session) -> list[CheckResult]:
    size = session.scalar(text("SELECT pg_database_size(current_database())")) or 0
    status = Status.WARN if size > DB_WARN_BYTES else Status.INFO
    detail = f"{size / 1024**2:.1f} MB / {DB_FREE_BYTES / 1024**2:.0f} MB"
    if size > DB_WARN_BYTES:
        detail += " (consider removing old transcripts)"
    return [CheckResult(Section.USAGE, "Database size", status, detail)]


CONSOLE_ONLY = [
    CheckResult(Section.USAGE, "Modal credits", Status.INFO, "check https://modal.com/settings/usage"),
    CheckResult(
        Section.USAGE, "YouTube Data API quota", Status.INFO, "about 2 units/day; check Google Cloud console"
    ),
]


# --- orchestration ------------------------------------------------------------------------------


def _guard(name: str, section: Section, check: Callable[[], list[CheckResult]]) -> list[CheckResult]:
    try:
        return check()
    except Exception as exc:  # noqa: BLE001 - one broken check must not hide the rest
        return [CheckResult(section, name, Status.FAIL, f"{type(exc).__name__}: {str(exc)[:120]}")]


def run_health_checks(
    *,
    secrets: Secrets,
    settings: AppSettings | None,
    session: Session | None,
    client: httpx.Client,
    now: datetime,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    db_ok = False
    if session is None:
        results.append(CheckResult(Section.CONNECTIVITY, "Database", Status.FAIL, "DATABASE_URL is not set"))
    else:
        db = _guard("Database", Section.CONNECTIVITY, lambda: check_database(session))
        results += db
        db_ok = db[0].status is Status.OK

    models = settings.llm.models if settings else []
    first_channel = (
        next((c.channel_id for c in settings.channels if c.channel_id), None) if settings else None
    )
    results += _guard(
        "LINE", Section.CONNECTIVITY, lambda: check_line(_secret(secrets.line_channel_access_token), client)
    )
    results += _guard(
        "OpenRouter",
        Section.CONNECTIVITY,
        lambda: check_openrouter(_secret(secrets.openrouter_api_key), models, client),
    )
    results += _guard(
        "YouTube Data API",
        Section.CONNECTIVITY,
        lambda: check_youtube_api(_secret(secrets.youtube_api_key), first_channel, client),
    )
    results += check_heartbeat(_secret(secrets.heartbeat_url))
    if db_ok and session is not None:
        results += _guard(
            "Job runs", Section.CONNECTIVITY, lambda: check_job_runs(session, settings, now=now)
        )
        results += _guard("Database size", Section.USAGE, lambda: check_database_size(session))
    results += CONSOLE_ONLY
    return results
