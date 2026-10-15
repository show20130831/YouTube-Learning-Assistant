"""Blind human review of two runs, and the automatic concept check it validates.

Topic scores say nothing about whether a summary is right. For a few videos (the dataset's
``blind_subset``) a person compares the two runs' outputs without knowing which run wrote which:
the outputs are shown as X and Y in an order fixed by a seed. For each output the reviewer records

- each labelled concept: not mentioned, named only, explained correctly, or explained wrongly;
- whether the labelled main takeaway comes across;
- problems, by severity: E1 contradicts the transcript, E2 a specific claim the transcript does not
  support (naming a concept or paraphrasing is not one), E3 a missing concept or takeaway,
  E4 imprecise wording that does not change the meaning;

then which output is better. Judgments are saved after every video, so a review can be resumed.

The automatic concept check (``concept_mentioned``) only tells whether a concept is named in the
summary, key points or concept explanations; the review measures how often it agrees with a person.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel

from yla.eval.dataset import ConceptLabel, Dataset
from yla.eval.runner import Record
from yla.llm.schemas import VideoAnalysis

BLIND_SEED = 20261015


class ConceptState(StrEnum):
    NOT_MENTIONED = "not_mentioned"
    NAME_ONLY = "name_only"
    EXPLAINED = "explained"
    WRONG = "wrong"


class Severity(StrEnum):
    E1 = "E1"  # contradicts the transcript
    E2 = "E2"  # a specific claim the transcript does not support
    E3 = "E3"  # a labelled concept or the main takeaway is missing
    E4 = "E4"  # imprecise wording that does not change the meaning


class Preference(StrEnum):
    X = "X"
    Y = "Y"
    TIE = "tie"


class Issue(BaseModel):
    severity: Severity
    text: str  # the problematic statement, quoted from the output


class OutputJudgment(BaseModel):
    concepts: dict[str, ConceptState]  # concept term -> state
    takeaway_conveyed: bool
    issues: list[Issue] = []


class PairJudgment(BaseModel):
    video_id: str
    x_run: str
    y_run: str
    x: OutputJudgment
    y: OutputJudgment
    preference: Preference
    judged_at: datetime

    def for_run(self, run_id: str) -> OutputJudgment:
        return self.x if run_id == self.x_run else self.y

    def preferred_run(self) -> str | None:
        return {Preference.X: self.x_run, Preference.Y: self.y_run}.get(self.preference)


class BlindSession(BaseModel):
    run_a: str
    run_b: str
    seed: int = BLIND_SEED
    judgments: list[PairJudgment] = []

    def judged(self) -> set[str]:
        return {j.video_id for j in self.judgments}


def session_path(runs_dir: Path, run_a: str, run_b: str) -> Path:
    return runs_dir / "blind" / f"{run_a}__{run_b}.json"


def load_session(runs_dir: Path, run_a: str, run_b: str) -> BlindSession | None:
    path = session_path(runs_dir, run_a, run_b)
    if not path.exists():
        return None
    return BlindSession.model_validate_json(path.read_text(encoding="utf-8"))


def save_session(runs_dir: Path, session: BlindSession) -> None:
    path = session_path(runs_dir, session.run_a, session.run_b)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(session.model_dump(mode="json"), ensure_ascii=False, indent=1), encoding="utf-8"
    )


class PlannedPair(BaseModel):
    video_id: str
    x_run: str
    y_run: str


def plan_pairs(
    dataset: Dataset,
    run_a: str,
    records_a: Sequence[Record],
    run_b: str,
    records_b: Sequence[Record],
    *,
    seed: int = BLIND_SEED,
) -> list[PlannedPair]:
    """Blind-subset videos both runs answered, each with a seeded X/Y order."""
    answered_a = {r.video_id for r in records_a if r.status == "ok"}
    answered_b = {r.video_id for r in records_b if r.status == "ok"}
    rng = random.Random(seed)
    pairs = []
    for video_id in dataset.blind_subset:
        a_first = rng.random() < 0.5  # drawn for every video, so the order does not depend on failures
        if video_id in answered_a and video_id in answered_b:
            x, y = (run_a, run_b) if a_first else (run_b, run_a)
            pairs.append(PlannedPair(video_id=video_id, x_run=x, y_run=y))
    return pairs


def record_judgment(session: BlindSession, judgment: PairJudgment) -> BlindSession:
    others = [j for j in session.judgments if j.video_id != judgment.video_id]
    return session.model_copy(update={"judgments": [*others, judgment]})


def new_judgment(
    pair: PlannedPair,
    x: OutputJudgment,
    y: OutputJudgment,
    preference: Preference,
    now: datetime | None = None,
) -> PairJudgment:
    return PairJudgment(
        video_id=pair.video_id,
        x_run=pair.x_run,
        y_run=pair.y_run,
        x=x,
        y=y,
        preference=preference,
        judged_at=now or datetime.now(UTC),
    )


def _searchable(analysis: VideoAnalysis) -> str:
    # Explanations only: keywords and concept terms name a concept without explaining it.
    parts = [analysis.one_line_summary, *analysis.key_points, *(c.explanation for c in analysis.key_concepts)]
    return "\n".join(parts).casefold()


def concept_mentioned(analysis: VideoAnalysis, concept: ConceptLabel) -> bool:
    text = _searchable(analysis)
    return any(name.casefold() in text for name in concept.names())


def render_output(analysis: VideoAnalysis) -> str:
    lines = [f"一句話摘要：{analysis.one_line_summary}", "重點："]
    lines += [f"  - {p}" for p in analysis.key_points]
    if analysis.key_concepts:
        lines.append("核心概念：")
        lines += [f"  - {c.term}：{c.explanation}" for c in analysis.key_concepts]
    return "\n".join(lines)
