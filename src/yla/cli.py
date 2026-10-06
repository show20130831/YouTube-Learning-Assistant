"""Command-line entry point: ``yla <command>``."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from yla.config import DEFAULT_SETTINGS_PATH, load_settings

app = typer.Typer(help="YouTube Learning Assistant", no_args_is_help=True)


@app.callback()
def main() -> None:
    """YouTube Learning Assistant."""


@app.command("validate-config")
def validate_config(
    path: Annotated[Path, typer.Option("--path", "-p")] = DEFAULT_SETTINGS_PATH,
) -> None:
    """Check that the settings file is valid and print a short summary."""
    try:
        settings = load_settings(path)
    except (FileNotFoundError, ValidationError) as exc:
        typer.secho(f"Invalid config: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    enabled = settings.enabled_channels
    typer.echo(f"Config OK: {path}")
    typer.echo(f"  channels: {len(enabled)} enabled / {len(settings.channels)} total")
    typer.echo(f"  topics:   {len(settings.topics)}")
    typer.echo(f"  daily limit: {settings.user.daily_video_limit} videos, timezone {settings.user.timezone}")
