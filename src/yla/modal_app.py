"""Modal deployment: the two daily cron jobs, scheduled in Asia/Taipei time.

- 11:00 ``daily_pipeline``: discover new videos, grade content, analyse up to the daily limit.
- 12:00 ``send_digest``: push the LINE digest (at most once per day).

Caption fetching is deliberately not here: YouTube blocks caption requests from cloud IPs
(15/15 blocked in the Phase 0 test), so the local worker fetches them from a home network
before 11:00 and Modal only reads them from the database.

Deploy from the project root (needs ``config/settings.yaml`` and the ``yla-secrets`` secret):

    uv run modal secret create yla-secrets --from-dotenv .env
    uv run modal deploy src/yla/modal_app.py

Run a job now:  ``uv run modal run src/yla/modal_app.py::send_digest --dry-run``
"""

from __future__ import annotations

import logging
from pathlib import Path

import modal

# This module is imported both locally (to build and deploy) and inside the container, where it
# lives at /root/modal_app.py; project paths only exist locally.
PROJECT_ROOT = Path(__file__).resolve().parents[2] if modal.is_local() else Path("/root")
SETTINGS_FILE = PROJECT_ROOT / "config" / "settings.yaml"
REMOTE_ROOT = "/root/app"
TIMEZONE = "Asia/Taipei"

image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync(str(PROJECT_ROOT), extra_options="--no-dev")
    .env({"PYTHONUNBUFFERED": "1"})
    .workdir(REMOTE_ROOT)
    .add_local_python_source("yla")
)
# settings.yaml is personal and git-ignored; it is uploaded from the machine that deploys.
if SETTINGS_FILE.exists():
    image = image.add_local_file(SETTINGS_FILE, f"{REMOTE_ROOT}/config/settings.yaml")

app = modal.App("youtube-learning-assistant", image=image, secrets=[modal.Secret.from_name("yla-secrets")])


def _configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.function(schedule=modal.Cron("0 11 * * *", timezone=TIMEZONE), timeout=60 * 60)
def daily_pipeline(skip_discover: bool = False) -> None:
    """Same as ``yla run``; a failure exits non-zero so Modal marks the run failed."""
    from yla.cli import run_command

    _configure_logging()
    run_command(skip_discover=skip_discover)


@app.function(schedule=modal.Cron("0 12 * * *", timezone=TIMEZONE), timeout=10 * 60)
def send_digest(dry_run: bool = False) -> None:
    """Same as ``yla digest``. Runs even if the pipeline failed, so failures are reported on LINE."""
    from yla.cli import digest_command

    _configure_logging()
    digest_command(dry_run=dry_run)
