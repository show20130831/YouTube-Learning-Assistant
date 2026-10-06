"""Sync ``config/settings.yaml`` into the database.

Upserts the single user, channels, subscriptions and topics. Anything removed from the YAML is
disabled rather than deleted, so its history (videos, analyses, trends) is kept.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from yla.config import AppSettings
from yla.db.models import Channel, Subscription, Topic, User


@dataclass
class SyncReport:
    user_created: bool = False
    channels_added: list[str] = field(default_factory=list)
    channels_enabled: list[str] = field(default_factory=list)
    channels_disabled: list[str] = field(default_factory=list)
    channel_ids_set: list[str] = field(default_factory=list)
    topics_added: list[str] = field(default_factory=list)
    topics_disabled: list[str] = field(default_factory=list)


def get_or_create_user(session: Session, line_user_id: str | None) -> tuple[User, bool]:
    """Each deployment serves one user: reuse the first row if it exists."""
    user = session.scalars(select(User).order_by(User.id).limit(1)).first()
    if user is not None:
        return user, False
    user = User(line_user_id=line_user_id)
    session.add(user)
    session.flush()
    return user, True


def sync_config(session: Session, settings: AppSettings, line_user_id: str | None = None) -> SyncReport:
    """Apply settings to the database. The caller owns the transaction."""
    report = SyncReport()
    user, report.user_created = get_or_create_user(session, line_user_id)

    prefs = settings.user
    if line_user_id:
        user.line_user_id = line_user_id
    user.timezone = prefs.timezone
    user.notify_time = prefs.notify_time
    user.daily_video_limit = prefs.daily_video_limit
    user.include_shorts = prefs.include_shorts
    user.defer_max_days = prefs.defer_max_days

    _sync_channels(session, user, settings, report)
    _sync_topics(session, user, settings, report)
    session.flush()
    return report


def _sync_channels(session: Session, user: User, settings: AppSettings, report: SyncReport) -> None:
    wanted = {c.handle.casefold(): c for c in settings.channels}
    subs = {
        channel.handle.casefold(): (channel, sub)
        for channel, sub in session.execute(
            select(Channel, Subscription)
            .join(Subscription, Subscription.channel_id == Channel.id)
            .where(Subscription.user_id == user.id)
        )
    }

    for key, cfg in wanted.items():
        if key in subs:
            channel, sub = subs[key]
            if sub.enabled != cfg.enabled:
                sub.enabled = cfg.enabled
                (report.channels_enabled if cfg.enabled else report.channels_disabled).append(cfg.handle)
        else:
            existing = session.scalars(select(Channel).where(func.lower(Channel.handle) == key)).first()
            channel = existing or Channel(handle=cfg.handle)
            if existing is None:
                session.add(channel)
                session.flush()
            session.add(Subscription(user_id=user.id, channel_id=channel.id, enabled=cfg.enabled))
            report.channels_added.append(cfg.handle)

        # An explicit channel_id in the YAML always wins over a stored (e.g. resolved) one.
        if cfg.channel_id and channel.youtube_channel_id != cfg.channel_id:
            channel.youtube_channel_id = cfg.channel_id
            report.channel_ids_set.append(cfg.handle)

    for key, (channel, sub) in subs.items():
        if key not in wanted and sub.enabled:
            sub.enabled = False
            report.channels_disabled.append(channel.handle)


def _sync_topics(session: Session, user: User, settings: AppSettings, report: SyncReport) -> None:
    existing = {t.name: t for t in session.scalars(select(Topic).where(Topic.user_id == user.id))}
    wanted = {t.name for t in settings.topics}

    for cfg in settings.topics:
        topic = existing.get(cfg.name)
        if topic is None:
            session.add(Topic(user_id=user.id, name=cfg.name, aliases=list(cfg.aliases), enabled=True))
            report.topics_added.append(cfg.name)
        else:
            topic.aliases = list(cfg.aliases)
            topic.enabled = True

    for name, topic in existing.items():
        if name not in wanted and topic.enabled:
            topic.enabled = False
            report.topics_disabled.append(name)
