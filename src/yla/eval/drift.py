"""Weekly drift check: has the production model changed under the same name?

A free model behind a fixed ID can change without notice (a new version, another provider, a
reroute). The check re-runs the configured primary model over the labelled dataset and compares it
with a reference run of the same model, prompt and rules, so a difference can only come from the
model. Two kinds of change count as drift:

- identity: a different model answered than in the reference (no threshold needed);
- quality: final-topic micro-F1 or precision moved by more than the tolerated margin.

The margin is not chosen by hand: it comes from A/A comparisons (the same configuration run twice),
the largest difference identical runs produced (see ``eval/drift.yaml``). Anything larger is more
than run-to-run noise has shown so far.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from yla.eval.report import LoadedRun
from yla.eval.stats import Interval, f1_difference, paired_bootstrap, precision_difference

DEFAULT_DRIFT_CONFIG = Path("eval/drift.yaml")


class DriftConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reference_run: str
    margin: float
    margin_source: str  # where the margin comes from, quoted in every drift report


def load_drift_config(path: Path = DEFAULT_DRIFT_CONFIG) -> DriftConfig:
    return DriftConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@dataclass(frozen=True)
class DriftResult:
    reference_run: str
    run_id: str
    compared: int  # videos both runs answered
    f1: Interval | None
    precision: Interval | None
    margin: float
    reasons: list[str] = field(default_factory=list)  # why it counts as drift; empty if none
    notes: list[str] = field(default_factory=list)  # changes worth knowing that are not drift

    @property
    def drifted(self) -> bool:
        return bool(self.reasons)


def _answered(run: LoadedRun) -> dict[str, tuple[str | None, str | None]]:
    return {r.video_id: (r.answered_by, r.provider) for r in run.records if r.status == "ok"}


def compare_for_drift(reference: LoadedRun, current: LoadedRun, *, margin: float) -> DriftResult:
    common = sorted(set(reference.score.scores) & set(current.score.scores))
    result = DriftResult(reference.meta.run_id, current.meta.run_id, len(common), None, None, margin)
    if not common:
        return result

    ref_answers, cur_answers = _answered(reference), _answered(current)
    reasons: list[str] = []
    notes: list[str] = []
    for kind, index in (("model", 0), ("provider", 1)):
        before = {ref_answers[v][index] for v in common}
        after = {cur_answers[v][index] for v in common}
        if after - before:
            was, now = ", ".join(sorted(map(str, before))), ", ".join(sorted(map(str, after)))
            change = f"answering {kind} {was} -> {now}"
            # A new model behind the same ID is drift by itself; a new provider serving the same
            # model is worth knowing but shows up in the scores if it matters.
            (reasons if kind == "model" else notes).append(change)

    ref_counts = [reference.score.scores[v].counts("final") for v in common]
    cur_counts = [current.score.scores[v].counts("final") for v in common]
    f1 = paired_bootstrap(ref_counts, cur_counts, f1_difference)
    precision = paired_bootstrap(ref_counts, cur_counts, precision_difference)
    for name, interval in (("micro-F1", f1), ("precision", precision)):
        if abs(interval.estimate) > margin:
            reasons.append(f"{name} changed by {interval.estimate:+.3f} (tolerated margin {margin:.3f})")
    return DriftResult(
        reference.meta.run_id, current.meta.run_id, len(common), f1, precision, margin, reasons, notes
    )


def render_drift(result: DriftResult, config: DriftConfig) -> list[str]:
    lines = [
        f"# Drift check `{result.run_id}` against `{result.reference_run}`",
        "",
        f"- Videos compared (answered by both runs): {result.compared}",
        f"- Tolerated margin: {result.margin:.3f} ({config.margin_source})",
    ]
    if result.f1 and result.precision:
        lines += [
            f"- Final-topic micro-F1: {result.f1.estimate:+.3f} "
            f"(95% CI {result.f1.low:+.3f} to {result.f1.high:+.3f})",
            f"- Precision: {result.precision.estimate:+.3f} "
            f"(95% CI {result.precision.low:+.3f} to {result.precision.high:+.3f})",
        ]
    lines += ["", "**Drift**" if result.drifted else "**No drift detected**", ""]
    lines += [f"- {r}" for r in result.reasons]
    lines += [f"- Note: {n}" for n in result.notes]
    return lines
