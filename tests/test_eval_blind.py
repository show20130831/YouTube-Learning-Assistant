from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from typer.testing import CliRunner

from yla import cli
from yla.config import TopicConfig
from yla.eval.blind import (
    BlindSession,
    ConceptState,
    Issue,
    OutputJudgment,
    Preference,
    Severity,
    concept_mentioned,
    load_session,
    new_judgment,
    plan_pairs,
    record_judgment,
)
from yla.eval.dataset import Cache, CachedVideo, ConceptLabel, Dataset, Item, Labels, Split
from yla.eval.report import build_report, load_run
from yla.eval.runner import Record, create_run, new_meta, open_run, run_dataset
from yla.llm.client import ChatMessage, LLMResponse
from yla.llm.schemas import VideoAnalysis
from yla.youtube.transcripts import TranscriptStatus

NOW = datetime(2026, 10, 15, 10, 0, tzinfo=UTC)
TOPICS = [TopicConfig(name="AI Agent", aliases=["Agent"])]
CONCEPTS = [ConceptLabel(term="KV Cache", aliases=["鍵值快取"]), ConceptLabel(term="Prefill")]


def analysis(explanation: str, keywords: list[str] | None = None) -> dict[str, Any]:
    return {
        "one_line_summary": "介紹語言模型的生成加速。",
        "key_points": ["重點一", "重點二"],
        "key_concepts": [{"term": "KV Cache", "explanation": explanation}],
        "keywords": keywords or [],
        "matched_topics": [],
        "relevance_score": 30,
        "limitation": None,
    }


def labelled(video_id: str) -> Item:
    return Item(
        video_id=video_id,
        split=Split.TEST,
        channel="Example",
        title=f"Video {video_id}",
        labels=Labels(
            topics=[],
            concepts=CONCEPTS,
            main_takeaway="KV Cache 讓生成時不必重算前面的 token。",
            takeaway_evidence="把 V 跟 K 存下來這件事情 就叫做 KV Cache",
        ),
    )


def record(video_id: str, status: str = "ok") -> Record:
    return Record(
        video_id=video_id,
        attempt_round=1,
        started_at=NOW,
        status=status,
        latency_seconds=1,
        calls=1,
        attempts=[],
        raw=analysis("存下 key 與 value，避免重算。") if status == "ok" else None,
    )


def test_concept_check_reads_explanations_not_keywords() -> None:
    explained = VideoAnalysis.model_validate(analysis("存下鍵值快取以加速 Prefill 之後的解碼。"))
    assert concept_mentioned(explained, CONCEPTS[0])  # by its alias
    assert concept_mentioned(explained, CONCEPTS[1])
    named_only = VideoAnalysis.model_validate(analysis("加速生成。", keywords=["Prefill"]))
    assert not concept_mentioned(named_only, CONCEPTS[1])  # a keyword names, it does not explain


def test_pairs_are_seeded_and_skip_unanswered_videos() -> None:
    data = Dataset(
        version="v", topics=TOPICS, blind_subset=["a", "b", "c", "d"], items=[labelled(v) for v in "abcd"]
    )
    records_a = [record(v) for v in "abcd"]
    records_b = [record(v) for v in "abc"] + [record("d", status="timeout")]
    first = plan_pairs(data, "A", records_a, "B", records_b)
    assert [p.video_id for p in first] == ["a", "b", "c"]  # d was not answered by B
    assert first == plan_pairs(data, "A", records_a, "B", records_b)  # same seed, same order
    orders = {
        p.x_run for seed in range(20) for p in plan_pairs(data, "A", records_a, "B", records_b, seed=seed)
    }
    assert orders == {"A", "B"}  # both runs get shown first


def judgment(state: ConceptState, issues: list[Issue], conveyed: bool = True) -> OutputJudgment:
    return OutputJudgment(
        concepts={"KV Cache": state, "Prefill": ConceptState.NOT_MENTIONED},
        takeaway_conveyed=conveyed,
        issues=issues,
    )


class ScriptedLLM:
    def __init__(self, explanation: str) -> None:
        self.explanation = explanation

    def complete_json(
        self, messages: list[ChatMessage], *, schema: dict[str, Any], schema_name: str, model: str
    ) -> LLMResponse:
        return LLMResponse(data=analysis(self.explanation), model=model)


def make_runs(tmp_path: Path) -> tuple[Dataset, Path, Path]:
    data = Dataset(version="v", topics=TOPICS, blind_subset=["k1"], items=[labelled("k1")])
    cache = Cache(tmp_path / "cache")
    cache.put(
        CachedVideo(
            video_id="k1",
            channel="Example",
            title="Video k1",
            description="",
            transcript_status=TranscriptStatus.MANUAL,
            language="zh-TW",
            text="把 V 跟 K 存下來這件事情 就叫做 KV Cache\n先做 Prefill",
            source="database",
            fetched_at=NOW,
        )
    )
    runs = tmp_path / "runs"
    for run_id, explanation in (("cur", "KV Cache 存下 key 與 value。"), ("new", "加速生成。")):
        run = create_run(new_meta(run_id=run_id, model=f"{run_id}:free", dataset=data, temperature=0.2), runs)
        run_dataset(run, data, cache, ScriptedLLM(explanation))
    path = tmp_path / "dataset.yaml"
    path.write_text(yaml.safe_dump(data.model_dump(mode="json"), allow_unicode=True), encoding="utf-8")
    return data, runs, tmp_path / "cache"


def test_report_summarises_the_review_and_validates_the_concept_check(tmp_path: Path) -> None:
    data, runs, _ = make_runs(tmp_path)
    pair = plan_pairs(data, "cur", [record("k1")], "new", [record("k1")])[0]
    current = judgment(ConceptState.EXPLAINED, [Issue(severity=Severity.E4, text="用詞")])
    candidate = judgment(
        ConceptState.WRONG, [Issue(severity=Severity.E1, text="KV Cache 會改變模型參數")], conveyed=False
    )
    x, y = (current, candidate) if pair.x_run == "cur" else (candidate, current)
    session = record_judgment(
        BlindSession(run_a="cur", run_b="new"), new_judgment(pair, x, y, Preference.TIE)
    )

    titles = {"k1": "Video k1"}
    loaded = [load_run(open_run(r, runs), data, titles) for r in ("cur", "new")]
    report = build_report(data, loaded, blind=session)
    assert "### Blind human review" in report
    assert "| Main takeaway conveyed | 1/1 | 0/1 |" in report
    assert "| E1 issues | 0 | 1 |" in report
    assert "- B, `k1`, E1: KV Cache 會改變模型參數" in report
    assert "用詞" not in report  # E4 is only counted
    # person: KV Cache mentioned in both, Prefill in neither; the string check agrees on 3 of 4
    assert "on 4 concept judgments (mentioned or not): agreement 0.75" in report


def test_cli_review_is_required_before_comparing_and_can_be_done_blind(tmp_path: Path) -> None:
    _, runs, cache_dir = make_runs(tmp_path)
    paths = ["--dataset", str(tmp_path / "dataset.yaml"), "--cache", str(cache_dir), "--runs", str(runs)]
    runner = CliRunner()

    refused = runner.invoke(cli.app, ["eval", "report", "cur", "new", *paths])
    assert refused.exit_code == 1 and "yla eval blind cur new" in refused.output
    assert runner.invoke(cli.app, ["eval", "report", "cur", "new", "--no-blind", *paths]).exit_code == 0

    # per output: two concepts, takeaway, one problem then an empty line; then the preference
    answers = "c\nn\ny\nE2 一個沒有依據的數字\n\nm\nn\nn\n\nx\n"
    review = runner.invoke(cli.app, ["eval", "blind", "cur", "new", *paths], input=answers)
    assert review.exit_code == 0, review.output
    assert (
        "cur" not in review.output.split("Output X")[1].split("Judging")[0]
    )  # outputs are not labelled by run
    session = load_session(runs, "cur", "new")
    assert session is not None and len(session.judgments) == 1
    [saved] = session.judgments
    assert saved.x.issues[0].severity is Severity.E2 and saved.preference is Preference.X
    assert runner.invoke(cli.app, ["eval", "report", "cur", "new", *paths]).exit_code == 0
