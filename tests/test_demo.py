"""The demo must work right after `uv sync`: no accounts, network, database or settings file."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import respx
from typer.testing import CliRunner

from yla.cli import app
from yla.demo import run_demo
from yla.notify.line import utf16_length


@respx.mock  # any real HTTP request would fail the test
def test_demo_covers_every_outcome() -> None:
    result = run_demo(today=date(2026, 10, 7))
    overview, videos, footer = result.messages

    assert "發現新影片：4 部（另略過 Shorts 1 部）" in overview
    assert "完成分析：3 部" in overview
    for stars in ("摘要信心：⭐⭐⭐", "摘要信心：⭐⭐\n", "摘要信心：⭐\n"):
        assert stars in videos
    assert "⚠️ 目前沒有取得字幕" in videos
    assert "🚫 無法分析（1 部）" in footer and "Community Q&A Stream" in footer
    assert all(utf16_length(m) <= 5000 for m in result.messages)


def test_demo_shows_code_side_fixes() -> None:
    result = run_demo()
    _, videos, _ = result.messages
    # Repeated gloss merged (with spacing), wrapping quotes removed, half-width gloss converted.
    assert "AI Agent 在正式環境" in videos and "AI Agent（AI Agent）" not in videos
    assert "「影片說明" not in videos
    assert "檢索增強生成（Retrieval-Augmented Generation）流程" in videos
    # A mechanical 可能 prefix is gone from the topic-guess points.
    assert "- 將耗時的 LLM 呼叫移到背景任務" in videos
    # Topics the model missed were added by TopicMatcher.
    assert "🎯 相關主題：LangGraph、AI Agent" in videos

    explanation = "\n".join(result.explanation)
    assert "added by TopicMatcher: ['LangGraph', 'AI Agent']" in explanation
    assert "a Short (<= 180s)" in explanation
    assert "set by code, not by the model" in explanation


def test_demo_is_deterministic() -> None:
    day = date(2026, 10, 7)
    assert run_demo(today=day) == run_demo(today=day)


@respx.mock
def test_cli_demo_needs_no_configuration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # no .env and no config/settings.yaml here
    for name in ("DATABASE_URL", "OPENROUTER_API_KEY", "LINE_CHANNEL_ACCESS_TOKEN", "YOUTUBE_API_KEY"):
        monkeypatch.delenv(name, raising=False)

    result = CliRunner().invoke(app, ["demo", "--explain"])
    assert result.exit_code == 0, result.output
    assert "Demo mode" in result.output
    assert "LINE digest (3 message(s)):" in result.output
    assert "[demo0000001] Building Reliable AI Agents with LangGraph" in result.output
