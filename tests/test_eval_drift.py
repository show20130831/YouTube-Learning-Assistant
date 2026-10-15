from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from yla.config import AppSettings, TopicConfig
from yla.db.models import JobRun, JobStatus, User
from yla.eval.dataset import Cache, CachedVideo, Dataset, Item, Labels, Split
from yla.eval.drift import DriftConfig, compare_for_drift, render_drift
from yla.eval.report import LoadedRun, load_run
from yla.eval.runner import create_run, new_meta, run_dataset
from yla.llm.client import ChatMessage, LLMResponse
from yla.ops.alerts import trend_warnings
from yla.sync import sync_config
from yla.youtube.transcripts import TranscriptStatus

NOW = datetime(2026, 10, 18, 6, 0, tzinfo=UTC)
TOPICS = [TopicConfig(name="AI Agent"), TopicConfig(name="RAG")]
CONFIG = DriftConfig(reference_run="ref", margin=0.1, margin_source="A/A comparisons")


def reply(topics: list[str]) -> dict[str, Any]:
    return {
        "one_line_summary": "摘要。",
        "key_points": ["一", "二"],
        "key_concepts": [],
        "keywords": [],
        "matched_topics": topics,
        "relevance_score": 50,
        "limitation": None,
    }


class Model:
    """Answers each video with fixed topics, as a given model and provider."""

    def __init__(
        self, topics: dict[str, list[str]], answered_by: str | None = None, provider: str = "P"
    ) -> None:
        self.topics, self.answered_by, self.provider = topics, answered_by, provider

    def complete_json(
        self, messages: list[ChatMessage], *, schema: dict[str, Any], schema_name: str, model: str
    ) -> LLMResponse:
        title = next(t for t in self.topics if t in messages[-1]["content"])
        return LLMResponse(
            data=reply(self.topics[title]), model=self.answered_by or model, provider=self.provider
        )


VIDEOS = {f"Video {n}": ["AI Agent"] for n in range(6)}


def dataset() -> Dataset:
    return Dataset(
        version="v",
        topics=TOPICS,
        items=[
            Item(
                video_id=f"v{n}",
                split=Split.TEST,
                channel="c",
                title=f"Video {n}",
                labels=Labels(topics=["AI Agent"], evidence={"AI Agent": "q"}),
            )
            for n in range(6)
        ],
    )


def runs(tmp_path: Path, current: Model, *, only: set[str] | None = None) -> tuple[LoadedRun, LoadedRun]:
    data = dataset()
    cache = Cache(tmp_path / "cache")
    for item in data.items:
        cache.put(
            CachedVideo(
                video_id=item.video_id,
                channel="c",
                title=item.title,
                description="",
                transcript_status=TranscriptStatus.MANUAL,
                language="en",
                text="text",
                source="database",
                fetched_at=NOW,
            )
        )
    titles = {i.video_id: i.title for i in data.items}
    loaded = []
    for run_id, model, subset in (("ref", Model(VIDEOS), None), ("now", current, only)):
        run = create_run(
            new_meta(run_id=run_id, model="m:free", dataset=data, temperature=0.2), tmp_path / "runs"
        )
        run_dataset(run, data, cache, model, only=subset)
        loaded.append(load_run(run, data, titles))
    return loaded[0], loaded[1]


def test_same_behaviour_is_not_drift(tmp_path: Path) -> None:
    result = compare_for_drift(*runs(tmp_path, Model(VIDEOS)), margin=0.1)
    assert (result.compared, result.drifted, result.reasons) == (6, False, [])
    assert "**No drift detected**" in render_drift(result, CONFIG)


def test_a_drop_beyond_the_margin_is_drift(tmp_path: Path) -> None:
    worse = {title: ["AI Agent", "RAG"] for title in VIDEOS}  # every video gets a wrong extra topic
    result = compare_for_drift(*runs(tmp_path, Model(worse)), margin=0.1)
    assert result.drifted
    assert result.reasons[0].startswith("micro-F1 changed by -0.333")
    text = "\n".join(render_drift(result, CONFIG))
    assert "**Drift**" in text and "Tolerated margin: 0.100 (A/A comparisons)" in text


def test_another_model_answering_is_drift_but_another_provider_is_a_note(tmp_path: Path) -> None:
    rerouted = compare_for_drift(*runs(tmp_path / "a", Model(VIDEOS, answered_by="other/model")), margin=0.1)
    assert rerouted.reasons == ["answering model m:free -> other/model"]
    moved = compare_for_drift(*runs(tmp_path / "b", Model(VIDEOS, provider="Q")), margin=0.1)
    assert (moved.drifted, moved.notes) == (False, ["answering provider P -> Q"])


def test_drift_runs_only_the_reference_videos(tmp_path: Path) -> None:
    reference, current = runs(tmp_path, Model(VIDEOS), only={"v0", "v1"})
    assert set(current.score.scores) == {"v0", "v1"}
    assert compare_for_drift(reference, current, margin=0.1).compared == 2


def setup_user(session: Session) -> User:
    sync_config(session, AppSettings.model_validate({"channels": [{"handle": "LangChain"}]}))
    return session.query(User).one()


@pytest.mark.db
def test_digest_warns_after_a_drift_check_found_drift(session: Session) -> None:
    user = setup_user(session)
    session.add(
        JobRun(
            job_name="eval_drift",
            status=JobStatus.SUCCEEDED,
            stats={
                "drifted": True,
                "reasons": ["micro-F1 changed by -0.150 (tolerated margin 0.098)"],
                "run_id": "d1",
            },
        )
    )
    session.flush()
    warnings = [w for w in trend_warnings(session, user, now=datetime.now(UTC)) if "漂移" in w]
    assert warnings == [
        "⚠️ 每週漂移檢查：主模型的表現和基準不同（micro-F1 changed by -0.150 (tolerated margin 0.098)），"
        "模型可能已在同一名稱下更換，請查看 eval/runs/d1/drift.md。"
    ]


@pytest.mark.db
def test_no_warning_without_drift(session: Session) -> None:
    user = setup_user(session)
    session.add(JobRun(job_name="eval_drift", status=JobStatus.SUCCEEDED, stats={"drifted": False}))
    session.flush()
    assert not [w for w in trend_warnings(session, user, now=datetime.now(UTC)) if "漂移" in w]


@pytest.mark.db
def test_cli_skips_and_records_when_production_changed_model(
    engine: Any, test_database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import yaml
    from sqlalchemy import select, text
    from typer.testing import CliRunner

    from yla import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    runs(tmp_path, Model(VIDEOS))  # reference run "ref" made with m:free
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump(dataset().model_dump(mode="json")), encoding="utf-8"
    )
    (tmp_path / "drift.yaml").write_text(yaml.safe_dump(CONFIG.model_dump()), encoding="utf-8")
    (tmp_path / "settings.yaml").write_text(
        "channels: [{handle: LangChain}]\nllm: {primary_model: 'new:free'}\n", encoding="utf-8"
    )
    args = ["--config", "drift.yaml", "--settings", "settings.yaml", "--dataset", "dataset.yaml"]
    args += ["--cache", "cache", "--runs", "runs"]
    try:
        result = CliRunner().invoke(cli.app, ["eval", "drift", *args])
        assert result.exit_code == 0, result.output
        assert "set a new reference" in result.output
        with Session(engine) as db:
            job = db.scalars(select(JobRun).where(JobRun.job_name == "eval_drift")).one()
            assert job.status is JobStatus.SUCCEEDED and job.stats["completed"] is False
            assert job.finished_at is not None
    finally:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE job_runs RESTART IDENTITY CASCADE"))
