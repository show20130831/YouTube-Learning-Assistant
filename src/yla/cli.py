"""Command-line entry point: ``yla <command>``."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from yla.analysis.analyzer import Analyzer
from yla.config import DEFAULT_SETTINGS_PATH, AppSettings, Secrets, load_settings
from yla.db.models import User, Video
from yla.db.session import make_engine, session_scope
from yla.llm.client import OpenRouterClient
from yla.pipeline.discovery import DiscoveryReport, discover_videos
from yla.sync import sync_config
from yla.youtube.channels import resolve_missing_channel_ids
from yla.youtube.data_api import MetadataProvider
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
    _print_discovery(report, resolved.unresolved)
    return report


def _print_discovery(report: DiscoveryReport, unresolved: list[str]) -> None:
    typer.echo(f"Checked {report.channels_checked} channel feeds")
    typer.echo(f"  new videos:  {len(report.new_videos)}")
    typer.echo(f"  backfilled:  {report.backfilled} (first read of a channel; not analysed)")
    for handle, error in report.failed_channels.items():
        _warn(f"  failed: {handle}: {error}")
    for handle in unresolved:
        _warn(f"  skipped {handle}: no channel_id (add it to settings.yaml, then run sync-config)")


@app.command("demo")
def demo_command(
    explain: Annotated[bool, typer.Option("--explain", help="Also show what each step did")] = False,
) -> None:
    """Run the content pipeline on demo data: no accounts, network or database needed."""
    from yla.demo import run_demo

    result = run_demo()
    typer.secho(
        "Demo mode: a fictional channel and recorded LLM replies; no accounts, network or database.",
        fg=typer.colors.CYAN,
    )
    if explain:
        typer.echo("")
        for line in result.explanation:
            typer.echo(line)
    typer.echo("")
    typer.echo(f"LINE digest ({len(result.messages)} message(s)):")
    for i, message in enumerate(result.messages, 1):
        typer.echo(f"----- message {i}/{len(result.messages)} -----")
        typer.echo(message)


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
    from yla.worker.local_worker import run_local_worker

    provider = YouTubeTranscriptProvider()
    with _db_session() as session:
        if check:
            _check_network(session, provider, video_id)
            return

        user = _user_or_exit(session)
        resolved = resolve_missing_channel_ids(session, user.id, resolver=None)
        with httpx.Client(timeout=20) as client:
            report = run_local_worker(
                session,
                user,
                provider,
                DatabaseTranscriptSink(session),
                now=datetime.now(UTC),
                feed_fetcher=None if skip_discover else HttpFeedFetcher(client),
                metadata=_metadata_provider(client),
            )

    if report.discovery is not None:
        _print_discovery(report.discovery, resolved.unresolved)
    result = report.transcripts
    saved = list(result.saved.values())
    typer.echo(f"Fetched captions for {len(saved)} of {result.attempted} videos")
    typer.echo(
        f"  manual: {saved.count(TranscriptStatus.MANUAL)}, auto: {saved.count(TranscriptStatus.GENERATED)}"
    )
    typer.echo(f"  no captions: {len(result.no_captions)}")
    for vid, error in result.errors.items():
        _warn(f"  error {vid}: {error}")
    if result.blocked:
        _warn("YouTube is blocking caption requests from this network; videos will use title + description.")
        raise typer.Exit(code=2)


@app.command("status")
def status_command(
    days: Annotated[int, typer.Option(help="How many days of job runs to show")] = 3,
) -> None:
    """Show recent job runs (local worker, daily pipeline, digest) for troubleshooting."""
    from zoneinfo import ZoneInfo

    from yla.db.models import JobRun

    with _db_session() as session:
        user = _user_or_exit(session)
        tz = ZoneInfo(user.timezone)
        since = datetime.now(UTC) - timedelta(days=days)
        runs = session.scalars(
            select(JobRun).where(JobRun.started_at >= since).order_by(JobRun.started_at.desc())
        ).all()

    if not runs:
        typer.echo(f"No job runs in the last {days} day(s).")
        return
    for run in runs:
        when = run.started_at.astimezone(tz).strftime("%m/%d %H:%M")
        color = {"succeeded": typer.colors.GREEN, "failed": typer.colors.RED}.get(run.status.value)
        typer.secho(f"{when}  {run.job_name:<15} {run.status.value:<10}", fg=color, nl=False)
        typer.echo(f" {_summarize_stats(run.job_name, run.stats)}")
        if run.error:
            _warn(f"               {run.error.splitlines()[0][:150]}")


def _summarize_stats(job_name: str, stats: dict[str, Any]) -> str:
    feed_errors = len(stats.get("feed_errors") or {})
    feeds = f", RSS failed {feed_errors}" if feed_errors else ""
    if job_name == "local_worker":
        blocked = ", BLOCKED" if stats.get("blocked") else ""
        shorts = f", Shorts {stats['shorts_skipped']}" if stats.get("shorts_skipped") else ""
        return (
            f"[{stats.get('host', '?')}] new {stats.get('new_videos', '-')}, "
            f"captions {stats.get('saved', 0)}/{stats.get('attempted', 0)}{shorts}{feeds}{blocked}"
        )
    if job_name == "daily_pipeline":
        quota = ", QUOTA USED UP" if stats.get("quota_exhausted") else ""
        return (
            f"analysed {stats.get('analyzed', 0)} ({stats.get('llm_calls', 0)} LLM), "
            f"deferred {stats.get('deferred', 0)}, unavailable {stats.get('unavailable', 0)}{feeds}{quota}"
        )
    if job_name == "send_digest":
        return f"{stats.get('messages', 0)} message(s)"
    return ""


def _metadata_provider(client: httpx.Client) -> MetadataProvider | None:
    """The optional YouTube Data API; without a key videos are treated as ordinary videos."""
    from yla.youtube.data_api import YouTubeDataApi

    key = Secrets().youtube_api_key
    return YouTubeDataApi(key.get_secret_value(), client) if key and key.get_secret_value() else None


def _openrouter_key_or_exit(settings: AppSettings) -> str:
    api_key = Secrets().openrouter_api_key
    if api_key is None:
        raise _fail("OPENROUTER_API_KEY is not set (see .env.example)")
    if not settings.llm.models:
        raise _fail("No LLM model configured: set llm.primary_model in settings.yaml")
    return api_key.get_secret_value()


def _build_analyzer(settings: AppSettings, api_key: str, client: httpx.Client) -> Analyzer:
    return Analyzer(
        OpenRouterClient(api_key, client),
        settings.llm.models,
        settings.topics,
        max_input_chars=settings.llm.max_input_chars,
    )


@app.command("run")
def run_command(
    skip_discover: Annotated[bool, typer.Option("--skip-discover", help="Do not read RSS first")] = False,
    path: SettingsPath = DEFAULT_SETTINGS_PATH,
) -> None:
    """Daily pipeline: discover, grade content, analyse up to the daily limit, defer the rest."""
    from yla.pipeline.daily import run_daily_pipeline

    settings = _load_settings_or_exit(path)
    api_key = _openrouter_key_or_exit(settings)
    with _db_session() as session, httpx.Client(timeout=180) as client:
        user = _user_or_exit(session)
        report = run_daily_pipeline(
            session,
            user,
            settings,
            _build_analyzer(settings, api_key, client),
            now=datetime.now(UTC),
            feed_fetcher=None if skip_discover else HttpFeedFetcher(client),
            metadata=_metadata_provider(client),
        )

    typer.echo(f"Daily pipeline {report.run_id}")
    typer.echo(f"  new videos:  {report.discovered}")
    typer.echo(f"  analysed:    {len(report.analyzed)}  ({report.llm_calls} LLM requests)")
    typer.echo(f"  unavailable: {len(report.unavailable)}")
    if report.shorts_skipped or report.waiting:
        typer.echo(f"  skipped Shorts: {report.shorts_skipped}, waiting for streams: {report.waiting}")
    if report.metadata_error:
        _warn(f"  YouTube Data API unavailable: {report.metadata_error}")
    typer.echo(f"  deferred:    {len(report.deferred)}")
    if report.failed or report.expired:
        _warn(f"  failed: {len(report.failed)}, expired: {len(report.expired)}")
    for handle, error in report.feed_errors.items():
        _warn(f"  feed failed: {handle}: {error}")
    if report.quota_exhausted:
        _warn("  OpenRouter free quota appears used up; remaining videos were deferred to tomorrow.")


@app.command("digest")
def digest_command(
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the LINE messages without sending or saving")
    ] = False,
) -> None:
    """Send today's study digest to LINE (at most once per day)."""
    from yla.notify.line import LineNotifier
    from yla.pipeline.digest import collect_digest, send_digest

    secrets = Secrets()
    with _db_session() as session:
        user = _user_or_exit(session)
        now = datetime.now(UTC)
        if dry_run:
            from yla.notify.formatter import build_messages

            messages = build_messages(collect_digest(session, user, now=now))
            session.rollback()
            for i, message in enumerate(messages, 1):
                typer.echo(f"----- message {i}/{len(messages)} -----")
                typer.echo(message)
            return

        token, to = secrets.line_channel_access_token, user.line_user_id or secrets.line_user_id
        if token is None or not to:
            raise _fail("LINE_CHANNEL_ACCESS_TOKEN and LINE_USER_ID must be set (see .env.example)")
        with httpx.Client(timeout=30) as client:
            outcome = send_digest(session, user, LineNotifier(token.get_secret_value(), to, client), now=now)

    if outcome.status == "already_sent":
        typer.echo(f"Digest for {outcome.day} was already sent; nothing to do.")
    elif outcome.status == "failed":
        raise _fail(f"Digest for {outcome.day} could not be sent: {outcome.error}")
    else:
        typer.echo(f"Sent digest for {outcome.day} ({len(outcome.messages)} messages).")


@app.command("analyze")
def analyze_command(
    video_id: Annotated[str, typer.Argument(help="YouTube video ID already in the database")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Print the result without saving")] = False,
    raw: Annotated[bool, typer.Option("--raw", help="Also print the model's original JSON")] = False,
    path: SettingsPath = DEFAULT_SETTINGS_PATH,
) -> None:
    """Analyse one video with the LLM (uses stored captions, else title + description)."""
    from yla.analysis.analyzer import AnalysisFailed, store_analysis
    from yla.content.tiering import choose_content
    from yla.db.models import Channel
    from yla.youtube.transcripts import DatabaseTranscriptProvider

    settings = _load_settings_or_exit(path)
    api_key = _openrouter_key_or_exit(settings)

    with _db_session() as session, httpx.Client(timeout=180) as client:
        user = _user_or_exit(session)
        video = session.scalars(select(Video).where(Video.youtube_video_id == video_id)).first()
        if video is None:
            raise _fail(f"Video {video_id} is not in the database (run `yla discover` first)")
        channel = session.get(Channel, video.channel_id)
        tier = choose_content(
            DatabaseTranscriptProvider(session).get(video.id),
            video.title,
            video.description,
            min_description_chars=settings.content.min_description_chars,
        )
        if not tier.available:
            _warn(f"{video_id}: not enough text to summarise (unavailable)")
            raise typer.Exit(code=1)

        analyzer = _build_analyzer(settings, api_key, client)
        try:
            outcome = analyzer.analyze(
                title=video.title, channel=(channel.title or channel.handle) if channel else "", tier=tier
            )
        except AnalysisFailed as exc:
            raise _fail(f"All models failed after {exc.calls} requests: {exc}") from exc
        if not dry_run:
            store_analysis(session, video=video, user_id=user.id, outcome=outcome)

    result = outcome.analysis
    typer.echo(f"{video.title}")
    typer.echo(f"摘要依據：{tier.basis_label}　摘要信心：{tier.confidence_stars}")
    if tier.warning:
        typer.echo(tier.warning)
    typer.echo(f"一句話摘要：{result.one_line_summary}")
    for point in result.key_points:
        typer.echo(f"- {point}")
    typer.echo(f"關鍵字：{'、'.join(result.keywords)}")
    typer.echo(f"🎯 相關主題：{'、'.join(result.matched_topics) or '（無）'}（{result.relevance_score}）")
    if result.limitation:
        typer.echo(f"限制：{result.limitation}")
    typer.echo(f"[model {outcome.model}, {outcome.calls} request(s){', not saved' if dry_run else ''}]")
    for model, error in outcome.errors.items():
        _warn(f"  {model} failed first: {error}")
    if raw:
        import json

        typer.echo(json.dumps(outcome.raw, ensure_ascii=False, indent=2))


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
