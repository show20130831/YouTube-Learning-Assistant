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
    if report.channel_ids_set:
        typer.echo(f"  channel ids set: {', '.join(report.channel_ids_set)}")


@app.command("discover")
def discover_command() -> None:
    """Read channel RSS feeds and record new videos (no API key needed)."""
    from datetime import UTC, datetime

    import httpx
    from sqlalchemy import select

    from yla.db.models import User
    from yla.db.session import make_engine, session_scope
    from yla.pipeline.discovery import discover_videos
    from yla.youtube.channels import resolve_missing_channel_ids
    from yla.youtube.rss import HttpFeedFetcher

    secrets = Secrets()
    if secrets.database_url is None:
        raise _fail("DATABASE_URL is not set (see .env.example)")

    engine = make_engine(secrets.database_url.get_secret_value())
    try:
        with session_scope(engine) as session, httpx.Client(timeout=20) as client:
            user = session.scalars(select(User).order_by(User.id).limit(1)).first()
            if user is None:
                raise _fail("No user yet: run `yla sync-config` first")
            # No resolver yet: handle -> channel_id lookup via the Data API is a planned extension.
            resolved = resolve_missing_channel_ids(session, user.id, resolver=None)
            report = discover_videos(session, user.id, HttpFeedFetcher(client), now=datetime.now(UTC))
    finally:
        engine.dispose()

    typer.echo(f"Checked {report.channels_checked} channel feeds")
    typer.echo(f"  new videos:  {len(report.new_videos)}")
    typer.echo(f"  backfilled:  {report.backfilled} (first read of a channel; not analysed)")
    for handle, error in report.failed_channels.items():
        typer.secho(f"  failed: {handle}: {error}", fg=typer.colors.YELLOW)
    for handle in resolved.unresolved:
        typer.secho(
            f"  skipped {handle}: no channel_id (add it to settings.yaml, then run sync-config)",
            fg=typer.colors.YELLOW,
        )
