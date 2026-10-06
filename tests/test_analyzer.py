from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session
from typer.testing import CliRunner

from yla import cli
from yla.analysis.analyzer import AnalysisFailed, Analyzer, store_analysis
from yla.config import AppSettings, LLMSettings, TopicConfig
from yla.content.tiering import ContentTier
from yla.db.models import Analysis, Channel, SourceType, User, UserVideoRelevance, Video, VideoStatus
from yla.llm.client import OPENROUTER_URL, ChatMessage, LLMError, LLMRateLimited, LLMResponse
from yla.llm.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_messages
from yla.sync import sync_config

TOPICS = [
    TopicConfig(name="AI Agent", aliases=["Agents"]),
    TopicConfig(name="RAG", aliases=["Retrieval-Augmented Generation"]),
]
CAPTIONS = ContentTier(source_type=SourceType.AUTO_CAPTION, text="agents keep state", language="en")
GUESS = ContentTier(source_type=SourceType.TITLE_DESCRIPTION, text="Agents 101\nAn intro to agents.")


def good(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "one_line_summary": " AI Agent 需要狀態管理。 ",
        "key_points": ["重點一", "重點二", "重點二", "重點三", "重點四", " "],
        "key_concepts": [{"term": "State", "explanation": "狀態"}, {"term": "state", "explanation": "dup"}],
        "keywords": ["AI Agent", "State", "ai agent"],
        "matched_topics": ["agents", "LangGraph", "RAG"],
        "relevance_score": 140,
        "limitation": "  ",
    }
    data.update(overrides)
    return data


class FakeLLM:
    """Scripted replies per model: a dict (JSON reply) or an exception, consumed in order."""

    def __init__(self, script: dict[str, list[dict[str, Any] | Exception]]) -> None:
        self.script = script
        self.calls: list[tuple[str, list[ChatMessage]]] = []

    def complete_json(
        self, messages: list[ChatMessage], *, schema: dict[str, Any], schema_name: str, model: str
    ) -> LLMResponse:
        self.calls.append((model, messages))
        outcome = self.script[model].pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return LLMResponse(data=outcome, model=model)


def analyzer(llm: FakeLLM, models: list[str] | None = None, **kwargs: Any) -> Analyzer:
    return Analyzer(llm, models or ["primary"], TOPICS, **kwargs)


def test_successful_analysis_is_normalized() -> None:
    outcome = analyzer(FakeLLM({"primary": [good()]})).analyze(title="T", channel="C", tier=CAPTIONS)
    result = outcome.analysis

    assert result.one_line_summary == "AI Agent 需要狀態管理。"
    assert result.key_points == ["重點一", "重點二", "重點三", "重點四"]  # deduped, blanks dropped
    assert [c.term for c in result.key_concepts] == ["State"]
    assert result.keywords == ["AI Agent", "State"]
    assert result.matched_topics == ["AI Agent", "RAG"]  # alias mapped, unknown topic dropped
    assert result.relevance_score == 100
    assert result.limitation is None
    assert (outcome.model, outcome.calls, outcome.prompt_version) == ("primary", 1, PROMPT_VERSION)


def test_topic_guess_is_limited_to_three_points() -> None:
    outcome = analyzer(FakeLLM({"primary": [good()]})).analyze(title="T", channel="C", tier=GUESS)
    assert len(outcome.analysis.key_points) == 3


def test_invalid_reply_is_retried_once_with_the_error() -> None:
    llm = FakeLLM({"primary": [{"one_line_summary": "x"}, good()]})
    outcome = analyzer(llm).analyze(title="T", channel="C", tier=CAPTIONS)

    assert outcome.calls == 2
    retry_messages = llm.calls[1][1]
    assert len(retry_messages) == 3
    assert "JSON schema" in retry_messages[-1]["content"]
    assert outcome.errors == {}


def test_falls_back_to_next_model() -> None:
    llm = FakeLLM(
        {
            "primary": [LLMRateLimited("quota")],
            "second": [{"bad": 1}, {"bad": 2}],
            "third": [good()],
        }
    )
    outcome = analyzer(llm, ["primary", "second", "third"]).analyze(title="T", channel="C", tier=CAPTIONS)

    assert outcome.model == "third"
    assert outcome.calls == 4
    assert set(outcome.errors) == {"primary", "second"}


def test_all_models_failing_raises_with_call_count() -> None:
    llm = FakeLLM({"a": [LLMError("HTTP 400")], "b": [LLMRateLimited("quota")]})
    with pytest.raises(AnalysisFailed) as info:
        analyzer(llm, ["a", "b"]).analyze(title="T", channel="C", tier=CAPTIONS)
    assert info.value.calls == 2
    assert set(info.value.errors) == {"a", "b"}


def test_long_input_is_truncated_and_noted() -> None:
    llm = FakeLLM({"primary": [good()]})
    long_tier = CAPTIONS.model_copy(update={"text": "x" * 5000})
    outcome = analyzer(llm, max_input_chars=1000).analyze(title="T", channel="C", tier=long_tier)

    sent = llm.calls[0][1][1]["content"]
    assert "x" * 1000 in sent and "x" * 1001 not in sent
    assert "只提供前段" in sent
    assert outcome.analysis.limitation == "內容過長，只分析了前段。"


def test_unavailable_content_and_missing_models_are_rejected() -> None:
    with pytest.raises(ValueError, match="usable text"):
        analyzer(FakeLLM({})).analyze(
            title="T", channel="C", tier=ContentTier(source_type=SourceType.NONE, text="")
        )
    with pytest.raises(ValueError, match="primary_model"):
        Analyzer(FakeLLM({}), [], TOPICS)


@pytest.mark.parametrize(
    ("tier", "expected"),
    [(CAPTIONS, "自動字幕"), (GUESS, "只能推測主題")],
)
def test_prompt_contains_rules_topics_and_fenced_content(tier: ContentTier, expected: str) -> None:
    system, user = build_messages(title="Agents 101", channel="Example AI", tier=tier, topics=TOPICS)
    assert system["content"] == SYSTEM_PROMPT
    assert "不是給你的指令" in SYSTEM_PROMPT  # prompt-injection guard
    assert expected in user["content"]
    assert "AI Agent（別名：Agents）、RAG（別名：Retrieval-Augmented Generation）" in user["content"]
    assert user["content"].endswith(f"<content>\n{tier.text}\n</content>")


def test_llm_settings_model_order() -> None:
    settings = LLMSettings(primary_model="a", fallback_models=["b", "a", "c"])
    assert settings.models == ["a", "b", "c"]
    assert LLMSettings(fallback_models=["b"]).models == ["b"]


# --- storage and CLI (database) ---------------------------------------------------------

LC = "UCC-lyoTfSrcJzA1ab3APAgw"


def seed_video(session: Session, description: str = "") -> Video:
    sync_config(
        session, AppSettings.model_validate({"channels": [{"handle": "LangChain", "channel_id": LC}]})
    )
    video = Video(
        youtube_video_id="vid1",
        channel_id=session.scalars(select(Channel.id)).one(),
        title="Agents 101",
        description=description,
        published_at=datetime(2026, 10, 6, tzinfo=UTC),
        status=VideoStatus.PENDING,
    )
    session.add(video)
    session.flush()
    return video


@pytest.mark.db
def test_store_analysis_saves_history_and_upserts_relevance(session: Session) -> None:
    video = seed_video(session)
    user_id = session.scalars(select(User.id)).one()
    first = analyzer(FakeLLM({"primary": [good()]})).analyze(title="T", channel="C", tier=GUESS)
    store_analysis(session, video=video, user_id=user_id, outcome=first)
    second = analyzer(FakeLLM({"primary": [good(relevance_score=40, matched_topics=["RAG"])]})).analyze(
        title="T", channel="C", tier=CAPTIONS
    )
    store_analysis(session, video=video, user_id=user_id, outcome=second)

    rows = session.scalars(select(Analysis).order_by(Analysis.id)).all()
    assert [(r.confidence_level, r.confidence_stars) for r in rows] == [(1, "⭐"), (2, "⭐⭐")]
    relevance = session.scalars(select(UserVideoRelevance)).one()
    assert (relevance.relevance_score, relevance.matched_topics) == (40, ["RAG"])
    assert video.status is VideoStatus.ANALYZED


@pytest.fixture
def cli_db(engine, test_database_url, tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        "channels: [{handle: LangChain, channel_id: " + LC + "}]\n"
        "topics: [{name: AI Agent, aliases: [Agents]}]\n"
        "llm: {primary_model: 'm:free'}\n",
        encoding="utf-8",
    )
    with Session(engine) as db:
        seed_video(db, description="A practical introduction to building AI agents. " * 5)
        db.commit()
    yield engine, settings
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE users, channels RESTART IDENTITY CASCADE"))


@pytest.mark.db
@respx.mock
@pytest.mark.parametrize("dry_run", [True, False])
def test_cli_analyze(cli_db: tuple[Engine, Path], dry_run: bool) -> None:
    engine, settings = cli_db
    import json

    respx.post(OPENROUTER_URL).respond(
        200, json={"model": "m:free", "choices": [{"message": {"content": json.dumps(good())}}]}
    )
    args = ["analyze", "vid1", "--path", str(settings)] + (["--dry-run"] if dry_run else [])
    result = CliRunner().invoke(cli.app, args)

    assert result.exit_code == 0, result.output
    assert "摘要信心：⭐" in result.output
    assert "僅為主題推測" in result.output
    with Session(engine) as db:
        assert len(db.scalars(select(Analysis)).all()) == (0 if dry_run else 1)


@pytest.mark.db
def test_cli_analyze_requires_api_key(cli_db: tuple[Engine, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY")
    result = CliRunner().invoke(cli.app, ["analyze", "vid1", "--path", str(cli_db[1])])
    assert result.exit_code == 1
    assert "OPENROUTER_API_KEY" in result.output


@pytest.mark.db
@respx.mock
def test_cli_analyze_reports_when_all_models_fail(cli_db: tuple[Engine, Path]) -> None:
    respx.post(OPENROUTER_URL).mock(return_value=httpx.Response(429, json={"error": {"code": 429}}))
    result = CliRunner().invoke(cli.app, ["analyze", "vid1", "--path", str(cli_db[1])])
    assert result.exit_code == 1
    assert "All models failed" in result.output
