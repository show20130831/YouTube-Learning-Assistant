"""Score an evaluation run against the labels.

Scoring always starts from the stored raw model output and re-applies the current
post-processing rules ("replay"). That makes rule changes measurable without new LLM requests,
and splits topic quality into layers, so an improvement can be traced to the model or the rules:

- ``title_only``: topic names and aliases found in the title, no LLM (the baseline to beat);
- ``llm_only``: the model's own ``matched_topics`` after normalisation;
- ``final``: what the product shows (LLM topics plus the deterministic matches).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import ValidationError

from yla.analysis.analyzer import reconcile_topics
from yla.analysis.checks import fix_count, residual_violations
from yla.analysis.topics import TopicMatcher
from yla.eval.dataset import Dataset, Item
from yla.eval.runner import Record
from yla.eval.stats import Counts
from yla.llm.schemas import VideoAnalysis, normalize

LAYERS = ("title_only", "llm_only", "final")


@dataclass(frozen=True)
class VideoScore:
    video_id: str
    expected: frozenset[str]
    topics: dict[str, frozenset[str]]  # layer -> predicted topics
    first_try_valid: bool | None  # None when the first request produced no reply at all
    fixes: int  # how many places the rules changed the model's text
    violations: list[str] = field(default_factory=list)

    def counts(self, layer: str) -> Counts:
        predicted = self.topics[layer]
        return Counts(
            tp=len(predicted & self.expected),
            fp=len(predicted - self.expected),
            fn=len(self.expected - predicted),
        )

    @property
    def rescued(self) -> frozenset[str]:
        """Topics the rules added on top of the model's own choice."""
        return self.topics["final"] - self.topics["llm_only"]


@dataclass(frozen=True)
class Replay:
    """One stored answer passed through the current rules again."""

    raw: VideoAnalysis  # what the model returned
    llm: VideoAnalysis  # after normalisation (model's own topics)
    final: VideoAnalysis  # what the product shows
    topic_guess: bool


def replay(record: Record, title: str, matcher: TopicMatcher) -> Replay:
    if record.raw is None:
        raise ValueError(f"{record.video_id}: no stored answer to replay")
    try:
        parsed = VideoAnalysis.model_validate(record.raw)
    except ValidationError as exc:  # recorded as ok, so this means the schema changed since
        raise ValueError(f"{record.video_id}: stored output no longer matches the schema") from exc
    topic_guess = record.source_type == "title_description"
    llm = normalize(parsed, topic_lookup=matcher.lookup, topic_guess=topic_guess)
    return Replay(parsed, llm, reconcile_topics(llm, title, matcher), topic_guess)


def score_record(
    record: Record, item: Item, title: str, matcher: TopicMatcher, topic_names: set[str]
) -> VideoScore:
    if record.raw is None or item.labels is None:
        raise ValueError("only answered, labelled videos can be scored")
    first = record.attempts[0][1] if record.attempts else None
    first_try_valid = None if first in (None, "timeout", "rate_limited", "error") else first == "ok"
    replayed = replay(record, title, matcher)
    parsed, llm, final, topic_guess = replayed.raw, replayed.llm, replayed.final, replayed.topic_guess
    return VideoScore(
        video_id=record.video_id,
        expected=frozenset(item.labels.topics),
        topics={
            "title_only": frozenset(matcher.from_text(title)),
            "llm_only": frozenset(llm.matched_topics),
            "final": frozenset(final.matched_topics),
        },
        first_try_valid=first_try_valid,
        fixes=fix_count(parsed, final),
        violations=residual_violations(final, topic_names, topic_guess=topic_guess),
    )


@dataclass(frozen=True)
class Availability:
    videos: int  # labelled videos with text that the run attempted
    first_round_ok: int
    first_round_timeouts: int
    ok: int  # after retries
    timeouts: int  # videos whose latest attempt timed out
    rate_limited: int
    persistent_timeouts: list[str]  # timed out in two or more rounds and never answered
    latencies: list[float]
    calls: int


@dataclass(frozen=True)
class RunScore:
    scores: dict[str, VideoScore]  # video_id -> score, answered videos only
    availability: Availability


def latest_by_video(records: Sequence[Record]) -> dict[str, list[Record]]:
    grouped: dict[str, list[Record]] = {}
    for record in records:
        grouped.setdefault(record.video_id, []).append(record)
    return grouped


def score_run(records: Sequence[Record], dataset: Dataset, titles: dict[str, str]) -> RunScore:
    matcher = TopicMatcher(dataset.topics)
    names = {t.name for t in dataset.topics}
    items = {i.video_id: i for i in dataset.labelled}
    grouped = latest_by_video([r for r in records if r.video_id in items and r.status != "no_text"])

    scores: dict[str, VideoScore] = {}
    first_ok = first_timeouts = timeouts = rate_limited = calls = 0
    persistent: list[str] = []
    latencies: list[float] = []
    for video_id, attempts in grouped.items():
        calls += sum(r.calls for r in attempts)
        first_ok += attempts[0].status == "ok"
        first_timeouts += attempts[0].status == "timeout"
        answered = next((r for r in attempts if r.status == "ok"), None)
        if answered is not None:
            latencies.append(answered.latency_seconds)
            scores[video_id] = score_record(answered, items[video_id], titles[video_id], matcher, names)
            continue
        timeouts += attempts[-1].status == "timeout"
        rate_limited += attempts[-1].status == "rate_limited"
        if sum(r.status == "timeout" for r in attempts) >= 2:
            persistent.append(video_id)
    return RunScore(
        scores=scores,
        availability=Availability(
            videos=len(grouped),
            first_round_ok=first_ok,
            first_round_timeouts=first_timeouts,
            ok=len(scores),
            timeouts=timeouts,
            rate_limited=rate_limited,
            persistent_timeouts=sorted(persistent),
            latencies=sorted(latencies),
            calls=calls,
        ),
    )


def percentile(values: Sequence[float], q: float) -> float | None:
    """Nearest-rank percentile of sorted values."""
    if not values:
        return None
    rank = max(1, min(len(values), math.ceil(q * len(values))))
    return values[rank - 1]
