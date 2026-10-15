"""Weekly drift check: has the production model changed under the same name?

A free model behind a fixed ID can change without notice (a new version, another provider, a
reroute). The check re-runs the configured primary model on the videos a reference run answered,
with the same prompt and rules, so a difference can only come from the model side. It reports three
kinds of change:

- identity drift: a different model answered than in the reference (no threshold needed);
- quality drift: final-topic micro-F1 or precision moved by more than its threshold;
- execution environment change: another upstream provider served the same model. Recorded and shown
  in the report title, but not alerted on by itself: free models move between providers routinely,
  and if it matters it shows up as quality drift.

Thresholds are not chosen by hand. They are computed on every check from the A/A pairs listed in
``eval/drift.yaml`` (the same configuration run twice): for each metric, the largest 95% CI bound of
the A/A difference. They are provisional: with few videos and few A/A pairs they estimate run-to-run
noise only roughly, and they change as A/A pairs are added.

Scope: this detects changes in the outputs for videos already in the reference. It cannot detect a
shift in the inputs themselves (new channels, new kinds of videos).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from yla.eval.report import LoadedRun
from yla.eval.stats import Interval, PairedStatistic, f1_difference, paired_bootstrap, precision_difference

DEFAULT_DRIFT_CONFIG = Path("eval/drift.yaml")
SCOPE_NOTE = (
    "Detects changes in the outputs for videos already in the reference only; it cannot detect a shift "
    "in the inputs (new channels or new kinds of videos)."
)


class DriftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_run: str
    # Pairs of runs with the same model, prompt and rules: the thresholds are computed from them.
    aa_pairs: list[tuple[str, str]]


def load_drift_config(path: Path = DEFAULT_DRIFT_CONFIG) -> DriftConfig:
    return DriftConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _paired(a: LoadedRun, b: LoadedRun, statistic: PairedStatistic) -> Interval | None:
    common = sorted(set(a.score.scores) & set(b.score.scores))
    if not common:
        return None
    return paired_bootstrap(
        [a.score.scores[v].counts("final") for v in common],
        [b.score.scores[v].counts("final") for v in common],
        statistic,
    )


@dataclass(frozen=True)
class Thresholds:
    f1: float
    precision: float
    # (run A, run B, F1 CI bound, precision CI bound) for every A/A pair, shown in the report
    pairs: list[tuple[str, str, float, float]]


def thresholds_from_aa(pairs: Sequence[tuple[LoadedRun, LoadedRun]]) -> Thresholds:
    """For each metric, the largest |95% CI bound| of the A/A differences."""
    rows = []
    for a, b in pairs:
        f1, precision = _paired(a, b, f1_difference), _paired(a, b, precision_difference)
        if f1 is None or precision is None:
            continue
        rows.append(
            (
                a.meta.run_id,
                b.meta.run_id,
                max(abs(f1.low), abs(f1.high)),
                max(abs(precision.low), abs(precision.high)),
            )
        )
    if not rows:
        raise ValueError("no A/A pair with videos answered by both runs: thresholds cannot be computed")
    return Thresholds(f1=max(r[2] for r in rows), precision=max(r[3] for r in rows), pairs=rows)


class Outcome(StrEnum):
    STABLE = "stable"
    DRIFTED = "drifted"


@dataclass(frozen=True)
class DriftResult:
    reference_run: str
    run_id: str
    compared: int  # videos both runs answered
    f1: Interval | None
    precision: Interval | None
    thresholds: Thresholds
    identity: list[str] = field(default_factory=list)  # a different model answered
    quality: list[str] = field(default_factory=list)  # a metric moved beyond its threshold
    environment: list[str] = field(default_factory=list)  # another provider served the model

    @property
    def outcome(self) -> Outcome:
        return Outcome.DRIFTED if self.identity or self.quality else Outcome.STABLE

    @property
    def reasons(self) -> list[str]:
        return [*self.identity, *self.quality]


def _changes(reference: LoadedRun, current: LoadedRun, videos: Sequence[str], index: int) -> list[str]:
    ref = {r.video_id: (r.answered_by, r.provider) for r in reference.records if r.status == "ok"}
    cur = {r.video_id: (r.answered_by, r.provider) for r in current.records if r.status == "ok"}
    before = {ref[v][index] for v in videos}
    after = {cur[v][index] for v in videos}
    if not after - before:
        return []
    kind = ("model", "provider")[index]
    was, now = ", ".join(sorted(map(str, before))), ", ".join(sorted(map(str, after)))
    return [f"answering {kind} {was} -> {now}"]


def compare_for_drift(reference: LoadedRun, current: LoadedRun, *, thresholds: Thresholds) -> DriftResult:
    common = sorted(set(reference.score.scores) & set(current.score.scores))
    if not common:
        return DriftResult(reference.meta.run_id, current.meta.run_id, 0, None, None, thresholds)
    f1 = _paired(reference, current, f1_difference)
    precision = _paired(reference, current, precision_difference)
    quality = [
        f"{name} changed by {interval.estimate:+.3f} (threshold {limit:.3f})"
        for name, interval, limit in (
            ("micro-F1", f1, thresholds.f1),
            ("precision", precision, thresholds.precision),
        )
        if interval is not None and abs(interval.estimate) > limit
    ]
    return DriftResult(
        reference.meta.run_id,
        current.meta.run_id,
        len(common),
        f1,
        precision,
        thresholds,
        identity=_changes(reference, current, common, 0),
        quality=quality,
        environment=_changes(reference, current, common, 1),
    )


def _headline(result: DriftResult) -> str:
    kinds = [k for k, found in (("identity", result.identity), ("quality", result.quality)) if found]
    head = f"Drift ({', '.join(kinds)})" if kinds else "Stable"
    if result.environment:
        head += "; execution environment changed"
    return head


def render_drift(result: DriftResult) -> list[str]:
    t = result.thresholds
    lines = [
        f"# Drift check `{result.run_id}` against `{result.reference_run}`: {_headline(result)}",
        "",
        f"- Videos compared (answered by both runs): {result.compared}",
    ]
    if result.f1 and result.precision:
        lines += [
            f"- Final-topic micro-F1: {result.f1.estimate:+.3f} (95% CI {result.f1.low:+.3f} to "
            f"{result.f1.high:+.3f}); threshold {t.f1:.3f}",
            f"- Precision: {result.precision.estimate:+.3f} (95% CI {result.precision.low:+.3f} to "
            f"{result.precision.high:+.3f}); threshold {t.precision:.3f}",
        ]
    lines += ["", "## Changes", ""]
    lines += [f"- Identity drift: {c}" for c in result.identity]
    lines += [f"- Quality drift: {c}" for c in result.quality]
    lines += [f"- Execution environment change: {c}" for c in result.environment]
    if not (result.identity or result.quality or result.environment):
        lines.append("- none")
    lines += [
        "",
        "## Thresholds (provisional)",
        "",
        "For each metric, the largest |95% CI bound| of the difference between two runs of the same "
        "configuration. Few videos and few A/A pairs make this a rough estimate of run-to-run noise.",
        "",
        "| A/A pair | F1 bound | Precision bound |",
        "|---|---|---|",
        *(f"| `{a}` vs `{b}` | {f1:.3f} | {p:.3f} |" for a, b, f1, p in t.pairs),
        "",
        f"Scope: {SCOPE_NOTE}",
    ]
    return lines
