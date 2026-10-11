"""Job run lifecycle and build/config identity.

Every job (local worker, daily pipeline, digest) is recorded in ``job_runs``. A run killed by its
platform (Modal function timeout, Windows task time limit) never gets to mark itself finished,
so when a job starts it first fails any run of the same job that is older than that limit:
past the limit the process has certainly been stopped, so no progress heartbeat is needed.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from yla.config import AppSettings
from yla.db.models import JobRun, JobStatus

# Hard limits enforced by the platform running each job, plus a little slack.
JOB_TIME_LIMITS = {
    "daily_pipeline": timedelta(minutes=60),  # Modal function timeout
    "send_digest": timedelta(minutes=10),  # Modal function timeout
    "local_worker": timedelta(minutes=60),  # Windows scheduled task time limit
}
GRACE = timedelta(minutes=5)
STALE_REASON = "stale_run_timeout: exceeded the platform time limit without finishing"


def expire_stale_runs(session: Session, job_name: str, *, now: datetime) -> int:
    limit = JOB_TIME_LIMITS.get(job_name, timedelta(hours=1)) + GRACE
    stale = session.scalars(
        select(JobRun).where(
            JobRun.job_name == job_name,
            JobRun.status == JobStatus.RUNNING,
            JobRun.started_at < now - limit,
        )
    ).all()
    for run in stale:
        run.status, run.error, run.finished_at = JobStatus.FAILED, STALE_REASON, now
    return len(stale)


def start_job(
    session: Session, job_name: str, *, now: datetime, stats: dict[str, Any] | None = None
) -> JobRun:
    """Expire stale runs of this job, then record and commit a new running one."""
    expired = expire_stale_runs(session, job_name, now=now)
    job = JobRun(job_name=job_name, status=JobStatus.RUNNING, stats={**(stats or {}), "git_sha": git_sha()})
    if expired:
        job.stats["expired_stale_runs"] = expired
    session.add(job)
    session.commit()
    return job


def config_fingerprint(settings: AppSettings) -> str:
    """Short hash of the (non-secret) settings, to tell whether the cloud runs the same config.

    settings.yaml holds no credentials, so the whole validated config is hashed.
    """
    canonical = settings.model_dump_json(exclude_defaults=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def git_sha() -> str:
    """Code version: from YLA_GIT_SHA (set at deploy time for Modal), else the local checkout."""
    if sha := os.environ.get("YLA_GIT_SHA"):
        return sha
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return f"{sha}-dirty" if dirty else sha
