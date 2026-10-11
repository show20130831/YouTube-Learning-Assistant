from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from yla.config import AppSettings
from yla.db.models import JobRun, JobStatus
from yla.ops.jobs import STALE_REASON, config_fingerprint, expire_stale_runs, git_sha, start_job
from yla.ops.logging import RunIdFilter, run_context

NOW = datetime(2026, 10, 11, 4, 0, tzinfo=UTC)


def settings(**overrides: Any) -> AppSettings:
    return AppSettings.model_validate({"channels": [{"handle": "LangChain"}], **overrides})


def test_fingerprint_is_stable_and_tracks_changes() -> None:
    assert config_fingerprint(settings()) == config_fingerprint(settings())
    assert config_fingerprint(settings()) != config_fingerprint(settings(user={"daily_video_limit": 8}))
    assert len(config_fingerprint(settings())) == 12


def test_git_sha_prefers_deploy_time_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YLA_GIT_SHA", "abc1234")
    assert git_sha() == "abc1234"


def test_git_sha_without_git(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("YLA_GIT_SHA", raising=False)
    monkeypatch.setenv("PATH", "")
    assert git_sha() == "unknown"


def test_run_id_is_added_to_log_records() -> None:
    record = logging.LogRecord("yla", logging.INFO, __file__, 1, "hello", None, None)
    RunIdFilter().filter(record)
    assert record.run_id == "-"  # type: ignore[attr-defined]
    with run_context(42):
        RunIdFilter().filter(record)
        assert record.run_id == "42"  # type: ignore[attr-defined]


@pytest.mark.db
def test_stale_runs_are_expired_only_past_the_platform_limit(session: Session) -> None:
    old = JobRun(
        job_name="send_digest", status=JobStatus.RUNNING, stats={}, started_at=NOW - timedelta(minutes=20)
    )
    recent = JobRun(
        job_name="send_digest", status=JobStatus.RUNNING, stats={}, started_at=NOW - timedelta(minutes=5)
    )
    other = JobRun(
        job_name="daily_pipeline", status=JobStatus.RUNNING, stats={}, started_at=NOW - timedelta(minutes=20)
    )
    session.add_all([old, recent, other])
    session.flush()

    assert expire_stale_runs(session, "send_digest", now=NOW) == 1
    assert (old.status, old.error, old.finished_at) == (JobStatus.FAILED, STALE_REASON, NOW)
    assert recent.status is JobStatus.RUNNING  # still within 10 min + grace
    assert other.status is JobStatus.RUNNING  # daily_pipeline allows 60 min


@pytest.mark.db
def test_start_job_cleans_up_and_records_version(session: Session, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("YLA_GIT_SHA", "abc1234")
    session.add(
        JobRun(
            job_name="local_worker", status=JobStatus.RUNNING, stats={}, started_at=NOW - timedelta(hours=2)
        )
    )
    session.flush()

    job = start_job(session, "local_worker", now=NOW, stats={"host": "pc"})
    assert job.status is JobStatus.RUNNING
    assert job.stats == {"host": "pc", "git_sha": "abc1234", "expired_stale_runs": 1}
    statuses = session.scalars(select(JobRun.status).where(JobRun.job_name == "local_worker")).all()
    assert sorted(s.value for s in statuses) == ["failed", "running"]
