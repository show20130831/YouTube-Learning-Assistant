from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest

from yla.config import TopicConfig
from yla.eval.dataset import Cache, CachedVideo, Dataset, Item, Labels, Split
from yla.eval.fetch import VideoInfo, fill_cache
from yla.eval.report import build_report, load_run
from yla.eval.runner import create_run, new_meta, run_dataset
from yla.eval.scoring import residual_violations, score_run
from yla.llm.client import ChatMessage, LLMRateLimited, LLMResponse, LLMTimeout
from yla.llm.schemas import VideoAnalysis
from yla.youtube.transcripts import TranscriptResult, TranscriptStatus

NOW = datetime(2026, 10, 13, 14, 0, tzinfo=UTC)
TOPICS = [
    TopicConfig(name="AI Agent", aliases=["Agent", "Agents"]),
    TopicConfig(name="RAG"),
    TopicConfig(name="LangChain"),
]


def reply(
    topics: list[str], *, keywords: list[str] | None = None, summary: str = "介紹工具調用的流程。"
) -> dict[str, Any]:
    return {
        "one_line_summary": summary,
        "key_points": ["重點一", "重點二"],
        "key_concepts": [{"term": "Tool Calling", "explanation": "讓模型呼叫外部工具。"}],
        "keywords": keywords or ["Tool Calling"],
        "matched_topics": topics,
        "relevance_score": 70,
        "limitation": None,
    }


class FakeLLM:
    """Answers by video title; an exception in the script is raised instead of answering."""

    def __init__(self, script: dict[str, list[dict[str, Any] | Exception]]) -> None:
        self.script = script
        self.calls = 0

    def complete_json(
        self, messages: list[ChatMessage], *, schema: dict[str, Any], schema_name: str, model: str
    ) -> LLMResponse:
        self.calls += 1
        title = next(t for t in self.script if t in messages[-1]["content"])
        step = self.script[title].pop(0)
        if isinstance(step, Exception):
            raise step
        return LLMResponse(data=step, model=model, usage={"total_tokens": 100}, provider="FakeProvider")


def item(video_id: str, title: str, topics: list[str]) -> Item:
    return Item(
        video_id=video_id,
        split=Split.TEST,
        channel="Example",
        title=title,
        labels=Labels(topics=topics, evidence={t: "quote" for t in topics}),
    )


def dataset(*items: Item) -> Dataset:
    return Dataset(version="test.1", topics=TOPICS, items=list(items))


def cache_video(
    cache: Cache, video_id: str, title: str, text: str = "We build an agent.\nIt calls tools."
) -> None:
    cache.put(
        CachedVideo(
            video_id=video_id,
            channel="Example",
            title=title,
            description="",
            transcript_status=TranscriptStatus.GENERATED,
            language="en",
            text=text,
            source="database",
            fetched_at=NOW,
        )
    )


@pytest.fixture
def no_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Evaluation must never touch the production database."""

    def refuse(*_: Any, **__: Any) -> None:
        raise AssertionError("evaluation tried to open a database connection")

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("sqlalchemy.create_engine", refuse)
    monkeypatch.setattr("yla.db.session.make_engine", refuse)


def test_run_score_and_report_without_a_database(tmp_path: Path, no_database: None) -> None:
    data = dataset(
        item("v1", "Building agents", ["AI Agent"]),
        item("v2", "LangChain retrieval", ["LangChain", "RAG"]),
        item("v3", "Cooking show", []),
    )
    cache = Cache(tmp_path / "cache")
    for i in data.items:
        cache_video(cache, i.video_id, i.title)
    llm = FakeLLM(
        {
            "Building agents": [reply([])],  # the model misses the topic; the title rescues it
            "LangChain retrieval": [reply(["RAG"])],  # title adds LangChain
            "Cooking show": [reply(["RAG"])],  # a wrong topic: costs precision
        }
    )
    run = create_run(
        new_meta(run_id="r1", model="m:free", dataset=data, temperature=0.2, now=NOW), tmp_path / "runs"
    )
    summary = run_dataset(run, data, cache, llm, now=lambda: NOW)
    assert (summary.ok, summary.failed) == (3, 0)

    titles = {i.video_id: i.title for i in data.items}
    score = score_run(run.records(), data, titles)
    v1 = score.scores["v1"]
    assert v1.topics["llm_only"] == frozenset()
    assert v1.topics["final"] == {"AI Agent"}
    assert v1.rescued == {"AI Agent"}
    assert score.scores["v3"].counts("final").fp == 1

    record = run.records()[0]
    assert (record.provider, record.usage, record.answered_by) == (
        "FakeProvider",
        {"total_tokens": 100},
        "m:free",
    )

    report = build_report(data, [load_run(run, data, titles)])
    assert "| final | 0.75 | 1.00 | 0.86 |" in report  # 3 TP, 1 FP, 0 FN
    assert "Rules added 2 topic(s)" in report


def test_resume_retries_failures_and_stops_on_rate_limit(tmp_path: Path, no_database: None) -> None:
    data = dataset(item("v1", "First video", ["AI Agent"]), item("v2", "Second video", ["RAG"]))
    cache = Cache(tmp_path / "cache")
    for i in data.items:
        cache_video(cache, i.video_id, i.title)
    llm = FakeLLM(
        {
            "First video": [LLMTimeout("slow"), reply(["AI Agent"])],
            "Second video": [LLMRateLimited("quota"), reply(["RAG"])],
        }
    )
    run = create_run(new_meta(run_id="r", model="m:free", dataset=data, temperature=0.2), tmp_path / "runs")

    first = run_dataset(run, data, cache, llm)
    assert first.stopped_on_rate_limit and first.failed == 2

    second = run_dataset(run, data, cache, llm)  # the next day
    assert (second.ok, second.skipped) == (2, 0)
    third = run_dataset(run, data, cache, llm)
    assert third.skipped == 2  # answered videos are never asked again

    availability = score_run(run.records(), data, {"v1": "First video", "v2": "Second video"}).availability
    assert (availability.videos, availability.first_round_ok, availability.ok) == (2, 0, 2)
    assert availability.first_round_timeouts == 1


def test_single_model_means_no_fallback(tmp_path: Path) -> None:
    data = dataset(item("v1", "Agents", ["AI Agent"]))
    cache = Cache(tmp_path / "cache")
    cache_video(cache, "v1", "Agents")
    llm = FakeLLM({"Agents": [LLMTimeout("slow")]})
    run = create_run(new_meta(run_id="r", model="m:free", dataset=data, temperature=0.2), tmp_path / "runs")
    run_dataset(run, data, cache, llm)
    [record] = run.records()
    assert record.status == "timeout"
    assert {model for model, _ in record.attempts} == {"m:free"}


def test_comparison_and_aa_noise(tmp_path: Path) -> None:
    data = dataset(*(item(f"v{n}", f"Video {n}", ["AI Agent"]) for n in range(6)))
    cache = Cache(tmp_path / "cache")
    for i in data.items:
        cache_video(cache, i.video_id, i.title, text="Some text.\nMore text.")
    titles = {i.video_id: i.title for i in data.items}
    runs = []
    for run_id, hit in (("a", 3), ("b", 6)):
        llm = FakeLLM({f"Video {n}": [reply(["AI Agent"] if n < hit else [])] for n in range(6)})
        run = create_run(
            new_meta(run_id=run_id, model=f"{run_id}:free", dataset=data, temperature=0.2), tmp_path
        )
        run_dataset(run, data, cache, llm)
        runs.append(load_run(run, data, titles))
    report = build_report(data, runs)
    assert "Paired comparison" in report
    assert "no decision is taken" in report

    same = build_report(data, [runs[0], runs[0]])
    assert "A/A comparison" in same
    assert "differs by up to 0.000" in same


def test_residual_violations_detect_what_the_rules_should_remove() -> None:
    names = {"AI Agent"}
    clean = VideoAnalysis.model_validate(reply(["AI Agent"]))
    assert residual_violations(clean, names) == []
    dirty = VideoAnalysis.model_validate(
        reply(["Unknown"], summary="「使用大型語言模型 (Large Language Model) 與 AI Agent（AI Agent）」")
    )
    assert set(residual_violations(dirty, names)) == {
        "half-width gloss",
        "repeated gloss",
        "wrapping quotes",
        "topic outside the list",
    }


class FakeProvider:
    def __init__(self, results: list[TranscriptResult]) -> None:
        self.results = results
        self.asked: list[str] = []

    def fetch(self, video_id: str) -> TranscriptResult:
        self.asked.append(video_id)
        return self.results.pop(0)


def test_fetch_respects_the_daily_caption_limit_and_stops_when_blocked(tmp_path: Path) -> None:
    data = dataset(*(item(f"v{n}", f"Video {n}", []) for n in range(4)))
    info = {f"v{n}": VideoInfo("Example", f"Video {n}", "") for n in range(4)}
    cache = Cache(tmp_path)
    ok = TranscriptResult(status=TranscriptStatus.MANUAL, language="en", text="hello")
    stored = {"v0": ok}  # already in the database: exported, not fetched
    provider = FakeProvider([ok, ok])
    tz = ZoneInfo("Asia/Taipei")

    report = fill_cache(
        data,
        cache,
        info=info,
        stored=stored,
        provider=provider,
        now=lambda: NOW,
        tz=tz,
        daily_limit=2,
        sleep=lambda _: None,
    )
    assert report.exported == ["v0"] and list(report.fetched) == ["v1", "v2"]
    assert report.left_for_later == ["v3"]
    assert cache.get("v1") is not None and cache.get("v1").source == "youtube"  # type: ignore[union-attr]

    blocked = FakeProvider([TranscriptResult(status=TranscriptStatus.BLOCKED)])
    tomorrow = NOW + timedelta(days=1)
    again = fill_cache(
        data,
        cache,
        info=info,
        stored=stored,
        provider=blocked,
        now=lambda: tomorrow,
        tz=tz,
        daily_limit=2,
        sleep=lambda _: None,
    )
    assert again.blocked and cache.get("v3") is None
