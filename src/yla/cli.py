"""Command-line entry point: ``yla <command>``."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import httpx
import typer
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from yla.config import DEFAULT_SETTINGS_PATH, AppSettings, Secrets, load_settings
from yla.db.models import User, Video
from yla.db.session import make_engine, session_scope
from yla.pipeline.discovery import DiscoveryReport, discover_videos
from yla.sync import sync_config
from yla.youtube.channels import resolve_missing_channel_ids
from yla.youtube.rss import HttpFeedFetcher
from yla.youtube.transcripts import (
    DatabaseTranscriptSink,
    TranscriptStatus,
    YouTubeTranscriptProvider,
)

app = typer.Typer(help="YouTube Learning Assistant", no_args_is_help=True)

SettingsPath = Annotated[Path, typer.Option("--path", "-p", help="Settings YAML file")]


@app.callback()
def main() -> None:
    """YouTube Learning Assistant."""


def _fail(message: str) -> typer.Exit:
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


def _warn(message: str) -> None:
    typer.secho(message, fg=typer.colors.YELLOW)


def _load_settings_or_exit(path: Path) -> AppSettings:
    try:
        return load_settings(path)
    except (FileNotFoundError, ValidationError) as exc:
        raise _fail(f"Invalid config: {exc}") from exc


@contextmanager
def _db_session() -> Iterator[Session]:
    secrets = Secrets()
    if secrets.database_url is None:
        raise _fail("DATABASE_URL is not set (see .env.example)")
    engine = make_engine(secrets.database_url.get_secret_value())
    try:
        with session_scope(engine) as session:
            yield session
    finally:
        engine.dispose()


def _user_or_exit(session: Session) -> User:
    user = session.scalars(select(User).order_by(User.id).limit(1)).first()
    if user is None:
        raise _fail("No user yet: run `yla sync-config` first")
    return user


def _discover(session: Session, user: User, client: httpx.Client) -> DiscoveryReport:
    # No resolver yet: handle -> channel_id lookup via the Data API is a planned extension.
    resolved = resolve_missing_channel_ids(session, user.id, resolver=None)
    report = discover_videos(session, user.id, HttpFeedFetcher(client), now=datetime.now(UTC))
    session.commit()

    typer.echo(f"Checked {report.channels_checked} channel feeds")
    typer.echo(f"  new videos:  {len(report.new_videos)}")
    typer.echo(f"  backfilled:  {report.backfilled} (first read of a channel; not analysed)")
    for handle, error in report.failed_channels.items():
        _warn(f"  failed: {handle}: {error}")
    for handle in resolved.unresolved:
        _warn(f"  skipped {handle}: no channel_id (add it to settings.yaml, then run sync-config)")
    return report


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
    settings = _load_settings_or_exit(path)
    with _db_session() as session:
        report = sync_config(session, settings, Secrets().line_user_id)

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
    with _db_session() as session, httpx.Client(timeout=20) as client:
        _discover(session, _user_or_exit(session), client)


@app.command("worker")
def worker_command(
    check: Annotated[
        bool, typer.Option("--check", help="Only test whether this network can fetch captions")
    ] = False,
    video_id: Annotated[str | None, typer.Option(help="Video to use for --check")] = None,
    skip_discover: Annotated[bool, typer.Option("--skip-discover", help="Do not read RSS first")] = False,
) -> None:
    """Local worker: discover new videos, then fetch captions from this (home) network."""
    from yla.worker.local_worker import fetch_transcripts

    provider = YouTubeTranscriptProvider()
    with _db_session() as session:
        if check:
            _check_network(session, provider, video_id)
            return

        user = _user_or_exit(session)
        if not skip_discover:
            with httpx.Client(timeout=20) as client:
                _discover(session, user, client)

        report = fetch_transcripts(
            session,
            user.id,
            provider,
            DatabaseTranscriptSink(session),
            now=datetime.now(UTC),
            limit=user.daily_video_limit * 2,
        )

    counts = {
        s: list(report.saved.values()).count(s) for s in (TranscriptStatus.MANUAL, TranscriptStatus.GENERATED)
    }
    typer.echo(f"Fetched captions for {len(report.saved)} of {report.attempted} videos")
    typer.echo(f"  manual: {counts[TranscriptStatus.MANUAL]}, auto: {counts[TranscriptStatus.GENERATED]}")
    typer.echo(f"  no captions: {len(report.no_captions)}")
    for vid, error in report.errors.items():
        _warn(f"  error {vid}: {error}")
    if report.blocked:
        _warn("YouTube is blocking caption requests from this network; videos will use title + description.")
        raise typer.Exit(code=2)


def _check_network(session: Session, provider: YouTubeTranscriptProvider, video_id: str | None) -> None:
    if video_id is None:
        video_id = session.scalars(select(Video.youtube_video_id).order_by(Video.published_at.desc())).first()
    if video_id is None:
        raise _fail("No videos in the database yet: run `yla discover` first, or pass --video-id")

    result = provider.fetch(video_id)
    typer.echo(
        f"Video {video_id}: {result.status.value}" + (f" ({result.language})" if result.language else "")
    )
    if result.status is TranscriptStatus.BLOCKED:
        _warn(f"Blocked: {result.error}. This network cannot fetch captions.")
        raise typer.Exit(code=2)
    if result.status is TranscriptStatus.ERROR:
        _warn(f"Error: {result.error}")
        raise typer.Exit(code=1)
    typer.echo("This network can reach YouTube captions.")
