"""Production LLM metrics over a time window, for ``yla metrics``.

Built from two records the daily pipeline writes: one ``llm_requests`` row per request (latency,
tokens, result, which model and provider actually answered) and the stored ``analyses``. It shows
trends for a person to judge rather than raising alerts: with 1-5 videos a day, a threshold on a
rate or a latency could not be justified. Only invariants are alerted on (see ``ops.alerts``):
an answer from a different model than the one requested, and a rule violation left in an output.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import date, datetime
from zoneinfo import ZoneInfo

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from yla.analysis.checks import fix_count, residual_violations
from yla.db.models import (
    Analysis,
    LLMRequest,
    LLMResult,
    SourceType,
    Topic,
    User,
    UserVideoRelevance,
    Video,
)
from yla.llm.schemas import Concept, VideoAnalysis

ROUTER_PREFIX = "openrouter/"  # router IDs answer with whatever model they picked, by design


def same_model(requested: str, answered: str | None) -> bool:
    """Whether a reply came from the model requested (OpenRouter may drop the ``:free`` suffix)."""
    if answered is None or requested.startswith(ROUTER_PREFIX):
        return True
    return requested.removesuffix(":free") == answered.removesuffix(":free")


def percentile(values: Sequence[int], q: float) -> int | None:
    """Nearest-rank percentile; with few values p95 is simply the slowest."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(1, min(len(ordered), math.ceil(q * len(ordered)))) - 1]


@dataclass(frozen=True)
class ModelStats:
    model: str
    requests: int
    results: dict[str, int]
    p50_ms: int | None
    p95_ms: int | None
    avg_prompt_tokens: float | None
    avg_completion_tokens: float | None
    rerouted: int
    providers: dict[str, int]


@dataclass
class DayStats:
    day: date
    analyses: int = 0
    stars: Counter[int] = field(default_factory=Counter)
    topics: list[int] = field(default_factory=list)  # matched topics per analysed video
    requests: int = 0
    videos_tried: int = 0  # videos the pipeline sent to the LLM
    fallbacks: int = 0  # answered, but not by the first model tried
    unanswered: int = 0  # no model answered
    timeouts: int = 0
    ok_latencies: list[int] = field(default_factory=list)
    rule_fixes: int = 0
    violations: int = 0

    @property
    def topics_per_video(self) -> float | None:
        return sum(self.topics) / len(self.topics) if self.topics else None


@dataclass(frozen=True)
class Metrics:
    since: datetime
    models: list[ModelStats]
    days: list[DayStats]
    violations: list[tuple[str, list[str]]]  # (YouTube video ID, problems)


def _model_stats(requests: Sequence[LLMRequest]) -> list[ModelStats]:
    by_model: dict[str, list[LLMRequest]] = defaultdict(list)
    for r in requests:
        by_model[r.model].append(r)
    stats = []
    for model, rows in sorted(by_model.items(), key=lambda kv: -len(kv[1])):
        ok = [r for r in rows if r.result is LLMResult.OK]
        prompt = [r.prompt_tokens for r in ok if r.prompt_tokens is not None]
        completion = [r.completion_tokens for r in ok if r.completion_tokens is not None]
        stats.append(
            ModelStats(
                model=model,
                requests=len(rows),
                results=dict(Counter(r.result.value for r in rows)),
                p50_ms=percentile([r.latency_ms for r in ok], 0.5),
                p95_ms=percentile([r.latency_ms for r in ok], 0.95),
                avg_prompt_tokens=sum(prompt) / len(prompt) if prompt else None,
                avg_completion_tokens=sum(completion) / len(completion) if completion else None,
                rerouted=sum(not same_model(r.model, r.answered_by) for r in ok),
                providers=dict(Counter(r.provider or "unknown" for r in ok)),
            )
        )
    return stats


def _final_analysis(analysis: Analysis, relevance: UserVideoRelevance | None) -> VideoAnalysis:
    return VideoAnalysis(
        one_line_summary=analysis.one_line_summary,
        key_points=list(analysis.key_points),
        key_concepts=[Concept(**c) for c in analysis.key_concepts],
        keywords=list(analysis.keywords),
        matched_topics=list(relevance.matched_topics) if relevance else [],
        relevance_score=relevance.relevance_score if relevance else 0,
        limitation=analysis.limitation,
    )


def collect_metrics(session: Session, user: User, *, since: datetime) -> Metrics:
    tz = ZoneInfo(user.timezone)
    days: dict[date, DayStats] = {}

    def day_of(moment: datetime) -> DayStats:
        local = moment.astimezone(tz).date()
        return days.setdefault(local, DayStats(local))

    requests = list(
        session.scalars(select(LLMRequest).where(LLMRequest.created_at >= since).order_by(LLMRequest.id))
    )
    per_video: dict[tuple[int | None, int | None], list[LLMRequest]] = defaultdict(list)
    for r in requests:
        per_video[(r.job_run_id, r.video_id)].append(r)
        day = day_of(r.created_at)
        day.requests += 1
        day.timeouts += r.result is LLMResult.TIMEOUT
        if r.result is LLMResult.OK:
            day.ok_latencies.append(r.latency_ms)
    for attempts in per_video.values():
        day = day_of(attempts[0].created_at)
        day.videos_tried += 1
        answered = next((r for r in attempts if r.result is LLMResult.OK), None)
        if answered is None:
            day.unanswered += 1
        elif answered.model != attempts[0].model:
            day.fallbacks += 1

    topic_names = set(session.scalars(select(Topic.name).where(Topic.user_id == user.id)))
    violations: list[tuple[str, list[str]]] = []
    analyses = session.execute(
        select(Analysis, Video.youtube_video_id, UserVideoRelevance)
        .join(Video, Video.id == Analysis.video_id)
        .outerjoin(
            UserVideoRelevance,
            (UserVideoRelevance.video_id == Analysis.video_id) & (UserVideoRelevance.user_id == user.id),
        )
        .where(Analysis.created_at >= since)
        .order_by(Analysis.id)
    ).all()
    for analysis, youtube_id, relevance in analyses:
        day = day_of(analysis.created_at)
        day.analyses += 1
        day.stars[analysis.confidence_level] += 1
        final = _final_analysis(analysis, relevance)
        day.topics.append(len(final.matched_topics))
        guess = analysis.source_type is SourceType.TITLE_DESCRIPTION
        if problems := residual_violations(final, topic_names, topic_guess=guess):
            day.violations += 1
            violations.append((youtube_id, problems))
        with suppress(ValidationError):  # stored before the current schema: nothing to compare
            day.rule_fixes += fix_count(VideoAnalysis.model_validate(analysis.raw_response), final)

    return Metrics(
        since=since,
        models=_model_stats(requests),
        days=[days[d] for d in sorted(days)],
        violations=violations,
    )


def _rate(part: int, whole: int) -> str:
    """A rate shown with its counts: with so few requests the denominator matters."""
    return "-" if whole == 0 else f"{part}/{whole} ({part / whole:.0%})"


def _ms(value: int | None) -> str:
    return "-" if value is None else f"{value / 1000:.0f}s"


def render_metrics(metrics: Metrics, *, tz: ZoneInfo) -> list[str]:
    since = metrics.since.astimezone(tz).strftime("%Y-%m-%d %H:%M")
    lines = [f"LLM requests since {since} (daily pipeline only)", ""]
    if not metrics.models:
        lines += ["  none recorded (requests are recorded since the llm_requests table was added)", ""]
    for m in metrics.models:
        results = ", ".join(f"{k} {v}" for k, v in sorted(m.results.items(), key=lambda kv: -kv[1]))
        tokens = (
            f"{m.avg_prompt_tokens:,.0f} in / {m.avg_completion_tokens:,.0f} out"
            if m.avg_prompt_tokens is not None and m.avg_completion_tokens is not None
            else "-"
        )
        providers = ", ".join(f"{k} {v}" for k, v in m.providers.items()) or "-"
        lines += [
            f"  {m.model}",
            f"    requests {m.requests}: {results}",
            f"    timeout rate {_rate(m.results.get('timeout', 0), m.requests)}, "
            f"format error rate {_rate(m.results.get('invalid', 0), m.requests)}",
            f"    latency p50 {_ms(m.p50_ms)}, p95 {_ms(m.p95_ms)} (answered only); avg tokens {tokens}",
            f"    providers: {providers}; answered by another model: {m.rerouted}",
        ]
    total = DayStats(metrics.since.date())
    for d in metrics.days:
        total.requests += d.requests
        total.videos_tried += d.videos_tried
        total.fallbacks += d.fallbacks
        total.unanswered += d.unanswered
        total.timeouts += d.timeouts
    answered = total.videos_tried - total.unanswered
    lines += [
        "",
        f"Whole window: fallback rate {_rate(total.fallbacks, answered)} of answered videos, "
        f"unanswered {_rate(total.unanswered, total.videos_tried)}, "
        f"timeout rate {_rate(total.timeouts, total.requests)} of requests.",
    ]
    lines += [
        "",
        "Per day          analysed  3★/2★/1★  topics/video  tried  fallback  unanswered  timeouts  "
        "p50   rule fixes  violations",
    ]
    for d in metrics.days:
        topics = "-" if d.topics_per_video is None else f"{d.topics_per_video:.1f}"
        stars = f"{d.stars[3]}/{d.stars[2]}/{d.stars[1]}"
        p50 = _ms(percentile(d.ok_latencies, 0.5))
        lines.append(
            f"  {d.day:%Y-%m-%d}  {d.analyses:>8}  {stars:>8}  {topics:>12}  {d.videos_tried:>5}  "
            f"{d.fallbacks:>8}  {d.unanswered:>10}  {d.timeouts:>8}  {p50:>4}  "
            f"{d.rule_fixes:>10}  {d.violations:>10}"
        )
    lines += [
        "",
        "Read as trends, not alerts: with 1-5 videos a day single days swing a lot. The evaluation found "
        "the same model attaches different topics to about a third of videos between identical runs.",
    ]
    if metrics.violations:
        lines += [
            "",
            "Outputs with problems (rule-guaranteed ones should never appear; the others are only "
            "asked for in the prompt):",
        ]
        lines += [f"  {video}: {', '.join(problems)}" for video, problems in metrics.violations]
    return lines
