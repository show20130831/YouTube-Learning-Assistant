"""Resolving channel handles (``@LangChain``) to channel IDs (``UC...``).

RSS needs the channel ID. Today users put ``channel_id`` in settings.yaml. The extension point
for automatic lookup is ``ChannelResolver``: the planned implementation calls the YouTube Data
API ``channels.list(part=id, forHandle=...)`` (1 quota unit) when ``YOUTUBE_API_KEY`` is set.
Until then no resolver is configured and channels without an ID are reported and skipped, so a
missing API key never blocks RSS, analysis or delivery.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from yla.db.models import Channel, Subscription


class ChannelResolver(Protocol):
    def resolve(self, handle: str) -> str | None:
        """Return the channel ID for a handle (without ``@``), or None if not found."""
        ...


@dataclass
class ResolveReport:
    resolved: dict[str, str] = field(default_factory=dict)
    unresolved: list[str] = field(default_factory=list)


def resolve_missing_channel_ids(
    session: Session, user_id: int, resolver: ChannelResolver | None
) -> ResolveReport:
    """Fill in missing IDs for the user's enabled channels. Without a resolver, only report them."""
    report = ResolveReport()
    missing = session.scalars(
        select(Channel)
        .join(Subscription, Subscription.channel_id == Channel.id)
        .where(
            Subscription.user_id == user_id,
            Subscription.enabled.is_(True),
            Channel.youtube_channel_id.is_(None),
        )
        .order_by(Channel.id)
    ).all()

    for channel in missing:
        channel_id = resolver.resolve(channel.handle) if resolver else None
        if channel_id:
            channel.youtube_channel_id = channel_id
            report.resolved[channel.handle] = channel_id
        else:
            report.unresolved.append(channel.handle)
    session.flush()
    return report
