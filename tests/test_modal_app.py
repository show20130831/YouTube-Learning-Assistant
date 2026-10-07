"""The Modal jobs are thin wrappers around the CLI commands; check they call the right ones."""

from __future__ import annotations

import pytest

from yla import cli, modal_app


def test_app_and_schedule_timezone() -> None:
    assert modal_app.app.name == "youtube-learning-assistant"
    assert modal_app.TIMEZONE == "Asia/Taipei"


def test_daily_pipeline_runs_the_run_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "run_command", lambda **kwargs: calls.append(kwargs))
    modal_app.daily_pipeline.local()
    modal_app.daily_pipeline.local(skip_discover=True)
    assert calls == [{"skip_discover": False}, {"skip_discover": True}]


def test_send_digest_runs_the_digest_command(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(cli, "digest_command", lambda **kwargs: calls.append(kwargs))
    modal_app.send_digest.local(dry_run=True)
    assert calls == [{"dry_run": True}]
