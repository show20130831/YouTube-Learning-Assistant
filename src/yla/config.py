"""Configuration loading.

Two sources, deliberately kept apart:
- ``Secrets``: credentials from environment variables / ``.env`` (never committed).
- ``AppSettings``: personal preferences from ``config/settings.yaml`` (channels, topics, limits).
"""

from __future__ import annotations

from datetime import time
from pathlib import Path
from typing import Self
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")


class Secrets(BaseSettings):
    """Credentials. All optional so demo mode runs without any account; components check what they need."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    database_url: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    youtube_api_key: SecretStr | None = None
    line_channel_access_token: SecretStr | None = None
    line_user_id: str | None = None


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class UserSettings(_Strict):
    timezone: str = "Asia/Taipei"
    notify_time: time = time(12, 0)
    daily_video_limit: int = Field(default=5, ge=1, le=10)
    include_shorts: bool = False
    defer_max_days: int = Field(default=3, ge=0, le=14)

    @field_validator("timezone")
    @classmethod
    def _valid_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone: {value!r}") from exc
        return value


class ChannelConfig(_Strict):
    handle: str = Field(min_length=1)
    # Optional for now. When missing, the channel is skipped until a ChannelResolver (planned:
    # YouTube Data API ``channels.list(forHandle=...)``) fills it in. See yla.youtube.channels.
    channel_id: str | None = Field(default=None, pattern=r"^UC[\w-]{22}$")
    enabled: bool = True

    @field_validator("handle")
    @classmethod
    def _strip_at(cls, value: str) -> str:
        handle = value.strip().removeprefix("@")
        if not handle:
            raise ValueError("handle must not be empty")
        return handle


class TopicConfig(_Strict):
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)

    def all_names(self) -> list[str]:
        return [self.name, *self.aliases]


class ContentSettings(_Strict):
    min_description_chars: int = Field(default=150, ge=0)
    long_video_threshold_minutes: int = Field(default=120, ge=1)
    max_chunks: int = Field(default=6, ge=1)


class LLMSettings(_Strict):
    primary_model: str | None = None
    fallback_models: list[str] = Field(default_factory=list)


class AppSettings(_Strict):
    user: UserSettings = Field(default_factory=UserSettings)
    channels: list[ChannelConfig] = Field(min_length=1)
    topics: list[TopicConfig] = Field(default_factory=list)
    content: ContentSettings = Field(default_factory=ContentSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @model_validator(mode="after")
    def _no_duplicates(self) -> Self:
        handles = [c.handle.casefold() for c in self.channels]
        if dupes := {h for h in handles if handles.count(h) > 1}:
            raise ValueError(f"duplicate channel handles: {sorted(dupes)}")
        ids = [c.channel_id for c in self.channels if c.channel_id]
        if dupes := {i for i in ids if ids.count(i) > 1}:
            raise ValueError(f"duplicate channel ids: {sorted(dupes)}")

        seen: dict[str, str] = {}
        for topic in self.topics:
            for term in topic.all_names():
                key = term.casefold()
                if key in seen and seen[key] != topic.name:
                    raise ValueError(f"term {term!r} used by both {seen[key]!r} and {topic.name!r}")
                seen[key] = topic.name
        return self

    @property
    def enabled_channels(self) -> list[ChannelConfig]:
        return [c for c in self.channels if c.enabled]


def load_settings(path: Path = DEFAULT_SETTINGS_PATH) -> AppSettings:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; copy config/settings.example.yaml to {path} and edit it")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return AppSettings.model_validate(data)
