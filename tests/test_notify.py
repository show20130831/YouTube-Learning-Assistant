from __future__ import annotations

import json
from datetime import date

import httpx
import pytest
import respx

from yla.db.models import SourceType
from yla.notify.formatter import DigestData, DigestVideo, build_messages
from yla.notify.line import PUSH_URL, LineNotifier, NotifyError, utf16_length

DAY = date(2026, 10, 7)


def video(n: int = 1, *, score: int = 50, level: int = 2, **overrides: object) -> DigestVideo:
    source = {3: SourceType.MANUAL_CAPTION, 2: SourceType.AUTO_CAPTION, 1: SourceType.TITLE_DESCRIPTION}[
        level
    ]
    data: dict[str, object] = {
        "title": f"Video {n}",
        "channel": "LangChain",
        "url": f"https://youtube.com/watch?v=v{n}",
        "source_type": source,
        "confidence_level": level,
        "one_line_summary": f"摘要 {n}",
        "key_points": ["重點一", "重點二"],
        "keywords": ["RAG", "Agent"],
        "matched_topics": ["AI Agent"],
        "relevance_score": score,
    }
    data.update(overrides)
    return DigestVideo(**data)  # type: ignore[arg-type]


def digest(videos: list[DigestVideo], **overrides: object) -> DigestData:
    data: dict[str, object] = {"day": DAY, "channels_tracked": 5, "new_videos": len(videos), "videos": videos}
    data.update(overrides)
    return DigestData(**data)  # type: ignore[arg-type]


# --- formatter --------------------------------------------------------------------------


def test_quiet_day_is_a_single_message() -> None:
    [message] = build_messages(digest([], trends=[("AI Agent", 3)], notices=["💻 今日未收到本機字幕"]))
    assert message.startswith("📺 今日沒有新影片｜2026/10/07\n\n💻 今日未收到本機字幕")
    assert "💻 今日未收到本機字幕" in message
    assert "1. AI Agent：3 次" in message


def test_overview_videos_and_footer() -> None:
    messages = build_messages(
        digest(
            [video(1, score=30), video(2, score=90)], unavailable=["Live｜X"], deferred=2, trends=[("RAG", 4)]
        )
    )
    overview, videos, footer = messages
    assert overview.splitlines()[:5] == [
        "📺 今日 YouTube 學習摘要｜2026/10/07",
        "",
        "今日追蹤頻道：5 個",
        "發現新影片：2 部",
        "完成分析：2 部",
    ]
    assert videos.index("Video 2｜LangChain") < videos.index("Video 1｜LangChain")  # by relevance
    assert videos.startswith("🎥 影片 1\n")
    assert "━━━━━━━━━━━━━━" in videos
    assert footer.split("\n\n") == [
        "🚫 無法分析（1 部）\n- Live｜X（文字內容不足）",
        "⏭️ 延後至明天分析（2 部）",
        "📈 近 7 天熱門主題\n1. RAG：4 次",
    ]


def test_caption_based_block() -> None:
    _, block = build_messages(digest([video(level=3)]))
    assert "摘要依據：人工字幕或完整逐字稿\n摘要信心：⭐⭐⭐\n🎯 相關主題：AI Agent" in block
    assert "一句話摘要：\n摘要 1\n\n重點：\n- 重點一\n- 重點二\n\n關鍵字：\nRAG、Agent" in block
    assert block.endswith("🔗 https://youtube.com/watch?v=v1")
    assert "⚠️" not in block


def test_topic_guess_block() -> None:
    _, block = build_messages(digest([video(level=1, matched_topics=[])]))
    assert "摘要信心：⭐\n\n⚠️ 目前沒有取得字幕" in block
    assert "可能主題：\n摘要 1\n\n可能涵蓋：\n- 重點一" in block
    assert "關鍵字" not in block and "🎯" not in block


def test_upgraded_video_is_labelled() -> None:
    _, block = build_messages(digest([video(upgraded=True)]))
    assert block.startswith("🎥 影片 1（🔄 已更新摘要）")


def test_notices_appear_in_overview() -> None:
    overview = build_messages(digest([video()], notices=["⚠️ 今日分析流程失敗：x"]))[0]
    assert overview.endswith("\n\n⚠️ 今日分析流程失敗：x")


def test_only_unavailable_videos() -> None:
    messages = build_messages(digest([], unavailable=["A｜B"]))
    assert len(messages) == 2 and messages[1].startswith("🚫 無法分析（1 部）")


def test_many_long_videos_fit_in_five_messages() -> None:
    long_points = ["很長的重點內容，" * 40] * 6
    videos = [video(i, score=100 - i, key_points=long_points) for i in range(30)]
    messages = build_messages(digest(videos, trends=[("RAG", 1)]))

    assert len(messages) == 5
    assert all(utf16_length(m) <= 5000 for m in messages)
    shown = sum(m.count("🎥 影片") for m in messages)
    assert f"…其餘 {30 - shown} 部未列出" in messages[3]
    assert messages[4].startswith("📈")


def test_single_huge_video_is_truncated_to_fit() -> None:
    messages = build_messages(digest([video(one_line_summary="字" * 20_000)]))
    assert all(utf16_length(m) <= 5000 for m in messages)


def test_huge_unavailable_list_is_truncated_to_fit() -> None:
    messages = build_messages(
        digest([video()], unavailable=[f"很長的標題 {i}｜頻道" * 5 for i in range(500)])
    )
    assert all(utf16_length(m) <= 5000 for m in messages)


def test_utf16_length_counts_surrogate_pairs() -> None:
    assert utf16_length("📺a") == 3


# --- LINE client ------------------------------------------------------------------------


@pytest.fixture
def notifier() -> LineNotifier:
    return LineNotifier("token", "U123", httpx.Client(), backoff_seconds=0)


@respx.mock
def test_push_sends_text_messages_with_retry_key(notifier: LineNotifier) -> None:
    route = respx.post(PUSH_URL).respond(200, json={})
    assert notifier.push(["a", "b"], retry_key="key-1") is True

    request = route.calls.last.request
    assert json.loads(request.content) == {
        "to": "U123",
        "messages": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
    }
    assert request.headers["Authorization"] == "Bearer token"
    assert request.headers["X-Line-Retry-Key"] == "key-1"


@respx.mock
def test_already_accepted_retry_key_is_not_an_error(notifier: LineNotifier) -> None:
    respx.post(PUSH_URL).respond(409, json={"message": "The retry key is already accepted"})
    assert notifier.push(["a"], retry_key="k") is False


@respx.mock
def test_server_errors_are_retried_with_the_same_key(notifier: LineNotifier) -> None:
    route = respx.post(PUSH_URL)
    route.side_effect = [httpx.Response(500), httpx.ConnectError("x"), httpx.Response(200, json={})]
    assert notifier.push(["a"], retry_key="k") is True
    assert {c.request.headers["X-Line-Retry-Key"] for c in route.calls} == {"k"}


@respx.mock
@pytest.mark.parametrize("status", [400, 401, 429])
def test_client_errors_raise(notifier: LineNotifier, status: int) -> None:
    route = respx.post(PUSH_URL).respond(status, json={"message": "nope"})
    with pytest.raises(NotifyError, match=str(status)):
        notifier.push(["a"], retry_key="k")
    assert route.call_count == 1


@respx.mock
def test_network_failure_raises_notify_error(notifier: LineNotifier) -> None:
    respx.post(PUSH_URL).side_effect = httpx.ConnectError("offline")
    with pytest.raises(NotifyError, match="network"):
        notifier.push(["a"], retry_key="k")


@pytest.mark.parametrize("messages", [[], ["x"] * 6, ["字" * 5001]])
def test_invalid_payloads_are_rejected_before_sending(notifier: LineNotifier, messages: list[str]) -> None:
    with pytest.raises(ValueError):
        notifier.push(messages, retry_key="k")


def test_quiet_day_with_problems_does_not_claim_all_is_well() -> None:
    [message] = build_messages(digest([], notices=["⚠️ RSS 失敗"]))
    assert "系統運作正常" not in message
    assert message == "📺 今日沒有新影片｜2026/10/07\n\n⚠️ RSS 失敗"


def test_quiet_day_without_problems_says_all_is_well() -> None:
    assert build_messages(digest([])) == ["📺 今日沒有新影片｜2026/10/07\n系統運作正常"]


def test_status_lines_show_with_or_without_problems() -> None:
    ok = build_messages(digest([], status=["💻 本機 worker：10:31 完成（字幕 0/0）"]))
    assert ok == ["📺 今日沒有新影片｜2026/10/07\n系統運作正常\n💻 本機 worker：10:31 完成（字幕 0/0）"]
    overview = build_messages(digest([video()], status=["💻 info"], notices=["⚠️ warn"]))[0]
    assert overview.endswith("\n\n💻 info\n⚠️ warn")
