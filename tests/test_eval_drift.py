from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy.orm import Session

from yla.config import AppSettings, TopicConfig
from yla.db.models import JobRun, JobStatus, User
from yla.eval.dataset import Cache, CachedVideo, Dataset, Item, Labels, Split
from yla.eval.drift import (
    DriftConfig,
    Outcome,
    Thresholds,
    compare_for_drift,
    render_drift,
    thresholds_from_aa,
)
from yla.eval.report import LoadedRun, load_run
from yla.eval.runner import create_run, new_meta, run_dataset
from yla.llm.client import ChatMessage, LLMResponse
from yla.ops.alerts import trend_warnings
from yla.sync import sync_config
from yla.youtube.transcripts import TranscriptStatus

NOW = datetime(2026, 10, 18, 6, 0, tzinfo=UTC)
TOPICS = [TopicConfig(name="AI Agent"), TopicConfig(name="RAG")]
CONFIG = DriftConfig(reference_run="ref", aa_pairs=[("ref", "now")])
LIMITS = Thresholds(f1=0.1, precision=0.12, pairs=[("a1", "a2", 0.1, 0.12)])


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


def test_same_behaviour_is_stable(tmp_path: Path) -> None:
    result = compare_for_drift(*runs(tmp_path, Model(VIDEOS)), thresholds=LIMITS)
    assert (result.compared, result.outcome, result.reasons, result.environment) == (
        6,
        Outcome.STABLE,
        [],
        [],
    )
    text = "\n".join(render_drift(result))
    assert text.startswith("# Drift check `now` against `ref`: Stable")
    assert "cannot detect a shift in the inputs" in text  # the scope is always stated


def test_a_drop_beyond_its_threshold_is_quality_drift(tmp_path: Path) -> None:
    worse = {title: ["AI Agent", "RAG"] for title in VIDEOS}  # every video gets a wrong extra topic
    result = compare_for_drift(*runs(tmp_path, Model(worse)), thresholds=LIMITS)
    assert result.outcome is Outcome.DRIFTED
    assert result.quality == [
        "micro-F1 changed by -0.333 (threshold 0.100)",
        "precision changed by -0.500 (threshold 0.120)",  # each metric has its own threshold
    ]
    text = "\n".join(render_drift(result))
    assert ": Drift (quality)" in text and "| `a1` vs `a2` | 0.100 | 0.120 |" in text


def test_identity_is_drift_and_provider_is_an_environment_change(tmp_path: Path) -> None:
    rerouted = compare_for_drift(
        *runs(tmp_path / "a", Model(VIDEOS, answered_by="other/model")), thresholds=LIMITS
    )
    assert (rerouted.outcome, rerouted.identity) == (
        Outcome.DRIFTED,
        ["answering model m:free -> other/model"],
    )
    moved = compare_for_drift(*runs(tmp_path / "b", Model(VIDEOS, provider="Q")), thresholds=LIMITS)
    assert (moved.outcome, moved.environment) == (Outcome.STABLE, ["answering provider P -> Q"])
    assert render_drift(moved)[0].endswith(": Stable; execution environment changed")  # shown in the title


def test_thresholds_are_computed_per_metric_from_aa_pairs(tmp_path: Path) -> None:
    same = runs(tmp_path / "a", Model(VIDEOS))
    one_video_differs = {**VIDEOS, "Video 0": ["AI Agent", "RAG"]}
    noisy = runs(tmp_path / "b", Model(one_video_differs))
    limits = thresholds_from_aa([same, noisy])
    assert round(limits.pairs[0][2], 3) == 0.0  # identical runs: no noise
    assert limits.f1 == max(p[2] for p in limits.pairs) > 0
    assert limits.precision == max(p[3] for p in limits.pairs) >= limits.f1  # here precision moves more
    with pytest.raises(ValueError, match="no A/A pair"):
        thresholds_from_aa([])


def test_drift_runs_only_the_reference_videos(tmp_path: Path) -> None:
    reference, current = runs(tmp_path, Model(VIDEOS), only={"v0", "v1"})
    assert set(current.score.scores) == {"v0", "v1"}
    assert compare_for_drift(reference, current, thresholds=LIMITS).compared == 2


def setup_user(session: Session) -> User:
    sync_config(session, AppSettings.model_validate({"channels": [{"handle": "LangChain"}]}))
    return session.query(User).one()


def drift_job(
    session: Session, days_ago: float, status: JobStatus = JobStatus.SUCCEEDED, **stats: Any
) -> None:
    session.add(
        JobRun(
            job_name="eval_drift",
            status=status,
            stats=stats,
            started_at=datetime.now(UTC) - timedelta(days=days_ago),
        )
    )
    session.flush()


def drift_warnings(session: Session, user: User) -> list[str]:
    return [w for w in trend_warnings(session, user, now=datetime.now(UTC)) if "漂移" in w]


@pytest.mark.db
def test_digest_warns_while_the_latest_completed_check_found_drift(session: Session) -> None:
    user = setup_user(session)
    drift_job(
        session,
        1,
        outcome="drifted",
        identity=[],
        quality=["micro-F1 changed by -0.150 (threshold 0.098)"],
        run_id="d1",
    )
    assert drift_warnings(session, user) == [
        "⚠️ 每週漂移檢查：主模型和基準不同（micro-F1 changed by -0.150 (threshold 0.098)），"
        "模型可能已在同一名稱下更換，請查看 eval/runs/d1/drift.md。"
    ]
    drift_job(session, 0, outcome="stable")  # the next check is fine: the warning clears at once
    assert drift_warnings(session, user) == []


@pytest.mark.db
def test_no_drift_warnings_before_the_first_check(session: Session) -> None:
    assert drift_warnings(session, setup_user(session)) == []


@pytest.mark.db
def test_a_check_that_stopped_completing_is_a_warning(session: Session) -> None:
    user = setup_user(session)
    drift_job(session, 10, outcome="stable")
    drift_job(session, 3, outcome="incomplete")  # stopped by the quota: not a completed check
    assert drift_warnings(session, user) == [
        "⚠️ 每週漂移檢查已 10 天沒有完成（上次被免費額度擋下而未完成），請確認每週排程與電腦是否開機。"
    ]


@pytest.mark.db
def test_a_skipped_check_asks_for_a_new_reference(session: Session) -> None:
    user = setup_user(session)
    drift_job(session, 1, outcome="skipped")
    assert drift_warnings(session, user) == [
        "⚠️ 每週漂移檢查從未完成（上次因主模型或 prompt 已更換而跳過，需在 eval/drift.yaml 設定新的基準），"
        "請確認每週排程與電腦是否開機。"
    ]


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
    runs(tmp_path, Model(VIDEOS))  # runs "ref" and "now", both made with m:free
    (tmp_path / "dataset.yaml").write_text(
        yaml.safe_dump(dataset().model_dump(mode="json")), encoding="utf-8"
    )
    (tmp_path / "drift.yaml").write_text(
        yaml.safe_dump({"reference_run": "ref", "aa_pairs": [["ref", "now"]]}), encoding="utf-8"
    )
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
            assert job.status is JobStatus.SUCCEEDED
            assert (job.stats["outcome"], job.stats["completed"]) == ("skipped", False)
            assert job.finished_at is not None
    finally:
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE job_runs RESTART IDENTITY CASCADE"))
