"""Turn a day's results into LINE text messages (Traditional Chinese).

Layout: an overview message, the videos (several per message, highest relevance first), and
a footer (unavailable videos, deferred count, 7-day topic trends). LINE allows 5 messages per
push and 5,000 characters per message; videos that do not fit are summarised as a count.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from yla.content.tiering import BASIS_LABEL, STARS, TOPIC_GUESS_WARNING
from yla.db.models import SourceType
from yla.notify.line import MAX_MESSAGES_PER_PUSH, MAX_TEXT_LENGTH, utf16_length

SEPARATOR = "━━━━━━━━━━━━━━"
# Headroom under LINE's limit for the "其餘 N 部未列出" line.
_SAFE_LENGTH = MAX_TEXT_LENGTH - 200


@dataclass(frozen=True)
class DigestVideo:
    title: str
    channel: str
    url: str
    source_type: SourceType
    confidence_level: int
    one_line_summary: str
    key_points: list[str]
    keywords: list[str]
    matched_topics: list[str]
    relevance_score: int
    upgraded: bool = False  # re-analysed with captions after an earlier ⭐ summary


@dataclass(frozen=True)
class DigestData:
    day: date
    channels_tracked: int
    new_videos: int
    videos: list[DigestVideo]
    unavailable: list[str] = field(default_factory=list)  # "title｜channel"
    deferred: int = 0
    trends: list[tuple[str, int]] = field(default_factory=list)
    notices: list[str] = field(default_factory=list)


def _video_block(index: int, video: DigestVideo) -> str:
    lines = [
        f"🎥 影片 {index}{'（🔄 已更新摘要）' if video.upgraded else ''}",
        "",
        "標題：",
        f"{video.title}｜{video.channel}",
        "",
        f"摘要依據：{BASIS_LABEL.get(video.source_type, '')}",
        f"摘要信心：{STARS.get(video.confidence_level, '')}",
    ]
    if video.matched_topics:
        lines.append(f"🎯 相關主題：{'、'.join(video.matched_topics)}")
    lines.append("")

    if video.source_type is SourceType.TITLE_DESCRIPTION:
        lines += [TOPIC_GUESS_WARNING, "", "可能主題：", video.one_line_summary]
        if video.key_points:
            lines += ["", "可能涵蓋：", *(f"- {p}" for p in video.key_points)]
    else:
        lines += ["一句話摘要：", video.one_line_summary, "", "重點：", *(f"- {p}" for p in video.key_points)]
        if video.keywords:
            lines += ["", "關鍵字：", "、".join(video.keywords)]
    lines += ["", f"🔗 {video.url}"]
    block = "\n".join(lines)
    # A single absurdly long block must still fit in one message.
    while utf16_length(block) > _SAFE_LENGTH:
        block = block[: len(block) - 200] + "…"
    return block


def _overview(data: DigestData) -> str:
    lines = [
        f"📺 今日 YouTube 學習摘要｜{data.day:%Y/%m/%d}",
        "",
        f"今日追蹤頻道：{data.channels_tracked} 個",
        f"發現新影片：{data.new_videos} 部",
        f"完成分析：{len(data.videos)} 部",
    ]
    if data.notices:
        lines += ["", *data.notices]
    return "\n".join(lines)


def _trend_lines(trends: list[tuple[str, int]]) -> list[str]:
    return ["📈 近 7 天熱門主題", *(f"{i}. {name}：{n} 次" for i, (name, n) in enumerate(trends, 1))]


def _footer(data: DigestData) -> str | None:
    sections: list[str] = []
    if data.unavailable:
        sections.append(
            "\n".join(
                [
                    f"🚫 無法分析（{len(data.unavailable)} 部）",
                    *(f"- {t}（文字內容不足）" for t in data.unavailable),
                ]
            )
        )
    if data.deferred:
        sections.append(f"⏭️ 延後至明天分析（{data.deferred} 部）")
    if data.trends:
        sections.append("\n".join(_trend_lines(data.trends)))
    if not sections:
        return None
    text = "\n\n".join(sections)
    while utf16_length(text) > _SAFE_LENGTH:  # e.g. a very long unavailable list
        text = text[: len(text) - 200] + "…"
    return text


def build_messages(data: DigestData) -> list[str]:
    """At most 5 messages, each within LINE's length limit."""
    if not data.videos and not data.unavailable:
        lines = [f"📺 今日沒有新影片｜{data.day:%Y/%m/%d}"]
        # Only claim everything is fine when nothing went wrong.
        lines += ["", *data.notices] if data.notices else ["系統運作正常"]
        if data.trends:
            lines += ["", *_trend_lines(data.trends)]
        return ["\n".join(lines)]

    overview = _overview(data)
    footer = _footer(data)
    slots = MAX_MESSAGES_PER_PUSH - 1 - (1 if footer else 0)

    ordered = sorted(data.videos, key=lambda v: v.relevance_score, reverse=True)
    blocks = [_video_block(i, v) for i, v in enumerate(ordered, 1)]

    chunks: list[list[str]] = []
    for block in blocks:
        joined = "\n\n".join([*chunks[-1], SEPARATOR, block]) if chunks else ""
        if chunks and utf16_length(joined) <= _SAFE_LENGTH:
            chunks[-1] += [SEPARATOR, block]
        elif len(chunks) < slots:
            chunks.append([block])
        else:
            break
    shown = sum(1 for chunk in chunks for part in chunk if part != SEPARATOR)
    if shown < len(blocks) and chunks:
        chunks[-1].append(f"…其餘 {len(blocks) - shown} 部未列出")

    messages = [overview, *("\n\n".join(chunk) for chunk in chunks)]
    if footer:
        messages.append(footer)
    return messages
