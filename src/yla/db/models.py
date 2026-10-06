"""SQLAlchemy models.

Analysis results are stored per video (shared), relevance per user, so adding users later
does not require re-analysing videos. Each deployment currently serves one user.
"""

from __future__ import annotations

from datetime import date, datetime, time
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    CheckConstraint,
    Date,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    SmallInteger,
    String,
    Text,
    Time,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Stable constraint names so Alembic autogenerate produces deterministic migrations.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}

JsonType = JSON().with_variant(JSONB(), "postgresql")


class VideoStatus(StrEnum):
    PENDING = "pending"
    WAITING = "waiting"  # upcoming premiere / live stream in progress
    DEFERRED = "deferred"  # over the daily limit or LLM failure; retried later
    ANALYZED = "analyzed"
    UNAVAILABLE = "unavailable"  # not enough text to summarise
    SKIPPED_SHORT = "skipped_short"
    SKIPPED_BACKFILL = "skipped_backfill"  # already in the feed when the channel was added
    FAILED = "failed"


class LiveStatus(StrEnum):
    NONE = "none"
    UPCOMING = "upcoming"
    LIVE = "live"


class SourceType(StrEnum):
    MANUAL_CAPTION = "manual_caption"
    AUTO_CAPTION = "auto_caption"
    TITLE_DESCRIPTION = "title_description"
    NONE = "none"


class FetchedBy(StrEnum):
    LOCAL = "local"
    MODAL = "modal"


class DigestStatus(StrEnum):
    PENDING = "pending"
    SENT = "sent"
    FAILED = "failed"


class JobStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


def _enum(enum_cls: type[StrEnum]) -> Enum:
    """Store enums as VARCHAR + CHECK constraint (easier to evolve than native PG enums)."""
    return Enum(
        enum_cls,
        name=enum_cls.__name__.lower(),
        native_enum=False,
        create_constraint=True,
        length=32,
        values_callable=lambda members: [m.value for m in members],
        validate_strings=True,
    )


def _now() -> Any:
    return mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    line_user_id: Mapped[str | None] = mapped_column(String(64), unique=True)
    timezone: Mapped[str] = mapped_column(String(64), default="Asia/Taipei")
    notify_time: Mapped[time] = mapped_column(Time, default=time(12, 0))
    daily_video_limit: Mapped[int] = mapped_column(SmallInteger, default=5)
    include_shorts: Mapped[bool] = mapped_column(default=False)
    defer_max_days: Mapped[int] = mapped_column(SmallInteger, default=3)
    created_at: Mapped[datetime] = _now()
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False
    )


class Channel(Base):
    __tablename__ = "channels"

    id: Mapped[int] = mapped_column(primary_key=True)
    handle: Mapped[str] = mapped_column(String(100))
    # Resolved from the handle via the YouTube Data API; null until resolved.
    youtube_channel_id: Mapped[str | None] = mapped_column(String(32), unique=True)
    title: Mapped[str | None] = mapped_column(String(200))
    created_at: Mapped[datetime] = _now()


# Handles are case-insensitive on YouTube (@LangChain == @langchain).
Index("uq_channels_handle_lower", func.lower(Channel.handle), unique=True)


class Subscription(Base):
    __tablename__ = "subscriptions"

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"), primary_key=True)
    enabled: Mapped[bool] = mapped_column(default=True)
    subscribed_at: Mapped[datetime] = _now()
    # Null means the feed has never been read: the first read marks existing videos as backfill.
    last_checked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Topic(Base):
    __tablename__ = "topics"
    __table_args__ = (UniqueConstraint("user_id", "name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    name: Mapped[str] = mapped_column(String(100))
    aliases: Mapped[list[str]] = mapped_column(JsonType, default=list)
    enabled: Mapped[bool] = mapped_column(default=True)


class Video(Base):
    __tablename__ = "videos"

    id: Mapped[int] = mapped_column(primary_key=True)
    youtube_video_id: Mapped[str] = mapped_column(String(16), unique=True)
    channel_id: Mapped[int] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text, default="")
    published_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    duration_seconds: Mapped[int | None] = mapped_column(Integer)
    live_status: Mapped[LiveStatus | None] = mapped_column(_enum(LiveStatus))
    is_short: Mapped[bool | None]
    status: Mapped[VideoStatus] = mapped_column(_enum(VideoStatus), default=VideoStatus.PENDING, index=True)
    attempts: Mapped[int] = mapped_column(SmallInteger, default=0)
    last_error: Mapped[str | None] = mapped_column(Text)
    discovered_at: Mapped[datetime] = _now()
    deferred_since: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Transcript(Base):
    __tablename__ = "transcripts"

    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), primary_key=True)
    source_type: Mapped[SourceType] = mapped_column(_enum(SourceType))
    language: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)
    char_count: Mapped[int] = mapped_column(Integer)
    fetched_by: Mapped[FetchedBy] = mapped_column(_enum(FetchedBy))
    fetched_at: Mapped[datetime] = _now()


class Analysis(Base):
    __tablename__ = "analyses"
    __table_args__ = (CheckConstraint("confidence_level BETWEEN 1 AND 3", name="confidence_level_range"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    model: Mapped[str] = mapped_column(String(200))
    prompt_version: Mapped[str] = mapped_column(String(32))
    source_type: Mapped[SourceType] = mapped_column(_enum(SourceType))
    confidence_level: Mapped[int] = mapped_column(SmallInteger)
    confidence_stars: Mapped[str] = mapped_column(String(8))
    limitation: Mapped[str | None] = mapped_column(Text)
    one_line_summary: Mapped[str] = mapped_column(Text)
    key_points: Mapped[list[str]] = mapped_column(JsonType)
    key_concepts: Mapped[list[dict[str, str]]] = mapped_column(JsonType)
    keywords: Mapped[list[str]] = mapped_column(JsonType)
    raw_response: Mapped[dict[str, Any]] = mapped_column(JsonType)
    created_at: Mapped[datetime] = _now()


class UserVideoRelevance(Base):
    __tablename__ = "user_video_relevance"
    __table_args__ = (CheckConstraint("relevance_score BETWEEN 0 AND 100", name="relevance_score_range"),)

    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), primary_key=True)
    relevance_score: Mapped[int] = mapped_column(SmallInteger)
    matched_topics: Mapped[list[str]] = mapped_column(JsonType, default=list)


class KeywordMention(Base):
    __tablename__ = "keyword_mentions"
    __table_args__ = (
        # A keyword counts once per video for trend statistics.
        UniqueConstraint("user_id", "video_id", "canonical_name"),
        Index("ix_keyword_mentions_user_date", "user_id", "mentioned_on"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    video_id: Mapped[int] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"))
    canonical_name: Mapped[str] = mapped_column(String(100))
    is_user_topic: Mapped[bool]
    mentioned_on: Mapped[date] = mapped_column(Date)


class Digest(Base):
    __tablename__ = "digests"
    __table_args__ = (UniqueConstraint("user_id", "digest_date"),)  # one push per user per day

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    digest_date: Mapped[date] = mapped_column(Date)
    status: Mapped[DigestStatus] = mapped_column(_enum(DigestStatus), default=DigestStatus.PENDING)
    message_text: Mapped[str | None] = mapped_column(Text)
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)


class JobRun(Base):
    __tablename__ = "job_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_name: Mapped[str] = mapped_column(String(64), index=True)
    started_at: Mapped[datetime] = _now()
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    status: Mapped[JobStatus] = mapped_column(_enum(JobStatus), default=JobStatus.RUNNING)
    stats: Mapped[dict[str, Any]] = mapped_column(JsonType, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
