"""Command-line entry point: ``yla <command>``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from yla.config import DEFAULT_SETTINGS_PATH, AppSettings, Secrets, load_settings

app = typer.Typer(help="YouTube Learning Assistant", no_args_is_help=True)

SettingsPath = Annotated[Path, typer.Option("--path", "-p", help="Settings YAML file")]


@app.callback()
def main() -> None:
    """YouTube Learning Assistant."""


def _fail(message: str) -> typer.Exit:
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def _load_settings_or_exit(path: Path) -> AppSettings:
    try:
        return load_settings(path)
    except (FileNotFoundError, ValidationError) as exc:
        raise _fail(f"Invalid config: {exc}") from exc


@app.command("validate-config")
def validate_config(path: SettingsPath = DEFAULT_SETTINGS_PATH) -> None:
    """Check that the settings file is valid and print a short summary."""
    settings = _load_settings_or_exit(path)
    enabled = settings.enabled_channels
    typer.echo(f"Config OK: {path}")
    typer.echo(f"  channels: {len(enabled)} enabled / {len(settings.channels)} total")
    typer.echo(f"  topics:   {len(settings.topics)}")
    typer.echo(f"  daily limit: {settings.user.daily_video_limit} videos, timezone {settings.user.timezone}")


@app.command("sync-config")
def sync_config_command(path: SettingsPath = DEFAULT_SETTINGS_PATH) -> None:
    """Write channels, topics and preferences from the settings file to the database."""
    from yla.db.session import make_engine, session_scope
    from yla.sync import sync_config

    settings = _load_settings_or_exit(path)
    secrets = Secrets()
    if secrets.database_url is None:
        raise _fail("DATABASE_URL is not set (see .env.example)")

    engine = make_engine(secrets.database_url.get_secret_value())
    try:
        with session_scope(engine) as session:
            report = sync_config(session, settings, secrets.line_user_id)
    finally:
        engine.dispose()

    typer.echo("Synced settings to database")
    if report.user_created:
        typer.echo("  user: created")
    typer.echo(
        f"  channels: {len(report.channels_added)} added, {len(report.channels_enabled)} enabled, "
        f"{len(report.channels_disabled)} disabled"
    )
    typer.echo(f"  topics:   {len(report.topics_added)} added, {len(report.topics_disabled)} disabled")
