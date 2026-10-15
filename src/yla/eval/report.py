"""Markdown report for one evaluation run, or a paired comparison of two.

Every number is reported with the conditions it was measured under (dataset version, number of
videos, prompt and rules versions). With a small dataset, a difference that is not significant
is not evidence of no difference, and the report says so instead of drawing conclusions.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from yla.analysis.analyzer import RULES_VERSION
from yla.analysis.topics import TopicMatcher
from yla.eval.blind import BlindSession, ConceptState, Severity, concept_mentioned
from yla.eval.dataset import Dataset, LabelStatus
from yla.eval.runner import Record, Run, RunMeta
from yla.eval.scoring import LAYERS, RunScore, VideoScore, percentile, replay, score_run
from yla.eval.stats import (
    Counts,
    clopper_pearson,
    f1_difference,
    mcnemar_exact_p,
    micro_prf,
    paired_bootstrap,
    paired_permutation_p,
    precision_difference,
)

MACRO_MIN_POSITIVES = 5  # topics with fewer labelled videos give an unstable F1


def _pct(value: float | None) -> str:
    return "-" if value is None else f"{value:.2f}"


def _signed(value: float) -> str:
    return f"{value:+.3f}"


@dataclass(frozen=True)
class LoadedRun:
    meta: RunMeta
    records: list[Record]
    score: RunScore
    titles: dict[str, str]


def load_run(run: Run, dataset: Dataset, titles: dict[str, str]) -> LoadedRun:
    records = run.records()
    return LoadedRun(run.meta(), records, score_run(records, dataset, titles), titles)


def _conditions(dataset: Dataset, runs: Sequence[LoadedRun]) -> list[str]:
    labelled = dataset.labelled
    status = Counter(i.labels.status for i in labelled if i.labels)
    splits = Counter(i.split.value for i in labelled)
    lines = [
        "## Conditions",
        "",
        f"- Dataset `{dataset.version}`: {len(labelled)} labelled videos "
        f"({', '.join(f'{k} {v}' for k, v in sorted(splits.items()))}); labels "
        f"{status.get(LabelStatus.CONFIRMED, 0)} confirmed by a person, "
        f"{status.get(LabelStatus.DRAFT, 0)} draft.",
        f"- Topic list: {len(dataset.topics)} topics, fingerprint `{dataset.topics_fingerprint}`"
        f"{f' (kept locally in `{dataset.topics_file}`)' if dataset.topics_file else ''}.",
        f"- Rules replayed: `{RULES_VERSION}` (scores are recomputed from the raw outputs).",
        "",
        "| Run | Model | Prompt | Rules at run time | Dataset at start | Temperature | Answered by "
        "| Providers |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for run in runs:
        answered = sorted({r.answered_by for r in run.records if r.answered_by})
        providers = sorted({r.provider for r in run.records if r.provider})
        m = run.meta
        lines.append(
            f"| `{m.run_id}` | `{m.model}` | `{m.prompt_version}` | `{m.rules_version}` | "
            f"`{m.dataset_version}` | {m.temperature} | "
            f"{', '.join(answered) or '-'} | {', '.join(providers) or 'not reported'} |"
        )
    return [*lines, ""]


def _availability(run: LoadedRun) -> list[str]:
    a = run.score.availability
    n = a.videos
    lines = [f"### Availability (all {n} attempted videos count)", ""]
    if not n:
        return [*lines, "No video attempted yet.", ""]
    low, high = clopper_pearson(a.first_round_timeouts, n)
    p50, p95 = percentile(a.latencies, 0.5), percentile(a.latencies, 0.95)
    lines += [
        "| Measure | Value |",
        "|---|---|",
        f"| Answered on the first attempt | {a.first_round_ok}/{n} |",
        f"| Answered after retries | {a.ok}/{n} |",
        f"| Timed out on the first attempt | {a.first_round_timeouts}/{n} (95% CI {low:.0%}-{high:.0%}) |",
        f"| Still unanswered: timeout / rate limited | {a.timeouts} / {a.rate_limited} |",
        f"| Timed out in two rounds | {', '.join(a.persistent_timeouts) or 'none'} |",
        f"| Latency p50 / p95 (answered) | {p50 or 0:.0f} s / {p95 or 0:.0f} s"
        f"{' (p95 is the slowest reply at this sample size)' if len(a.latencies) < 20 else ''} |",
        f"| Requests per video | {a.calls / n:.2f} |",
        "",
    ]
    return lines


def _topic_table(scores: Sequence[VideoScore]) -> list[str]:
    lines = ["| Layer | Precision | Recall | Micro-F1 |", "|---|---|---|---|"]
    for layer in LAYERS:
        prf = micro_prf([s.counts(layer) for s in scores])
        lines.append(f"| {layer} | {_pct(prf.precision)} | {_pct(prf.recall)} | {_pct(prf.f1)} |")
    rescued = [(s.rescued, s.expected) for s in scores]
    added = sum(len(r) for r, _ in rescued)
    correct = sum(len(r & e) for r, e in rescued)
    llm = micro_prf([s.counts("llm_only") for s in scores]).precision
    lines += [
        "",
        f"Rules added {added} topic(s) to the model's choice, {correct} of them correct "
        f"(rescue precision {_pct(correct / added if added else None)}, model's own precision {_pct(llm)}).",
        "",
    ]
    return lines


def _per_topic(scores: Sequence[VideoScore], topics: Sequence[str]) -> list[str]:
    lines = ["| Topic (final) | Labelled videos | TP | FP | FN |", "|---|---|---|---|---|"]
    macro: list[float] = []
    included: list[str] = []
    for topic in topics:
        tp = sum(topic in s.topics["final"] and topic in s.expected for s in scores)
        fp = sum(topic in s.topics["final"] and topic not in s.expected for s in scores)
        fn = sum(topic not in s.topics["final"] and topic in s.expected for s in scores)
        positives = tp + fn
        if positives >= MACRO_MIN_POSITIVES:
            macro.append(micro_prf([Counts(tp, fp, fn)]).f1 or 0.0)
            included.append(topic)
        lines.append(f"| {topic} | {positives} | {tp} | {fp} | {fn} |")
    total = sum((s.counts("final") for s in scores), Counts())
    lines += [
        f"| **All topics** | {total.tp + total.fn} | {total.tp} | {total.fp} | {total.fn} |",
        "",
        f"The totals give the final layer above: precision {total.tp}/{total.tp + total.fp}, "
        f"recall {total.tp}/{total.tp + total.fn}.",
    ]
    lines.append("")
    if macro:
        lines.append(
            f"Macro-F1 over only the topics with at least {MACRO_MIN_POSITIVES} labelled videos "
            f"({', '.join(included)}): {sum(macro) / len(macro):.2f}. Topics with fewer are left out, "
            "so this is not an average over all topics (descriptive only)."
        )
    else:
        lines.append(f"No topic has {MACRO_MIN_POSITIVES}+ labelled videos yet, so macro-F1 is not reported.")
    return [*lines, ""]


def _format(scores: Sequence[VideoScore]) -> list[str]:
    defined = [s.first_try_valid for s in scores if s.first_try_valid is not None]
    violations = Counter(v for s in scores for v in s.violations)
    fixes = sum(s.fixes for s in scores)
    detail = " - " + ", ".join(f"{k} {v}" for k, v in violations.items()) if violations else ""
    return [
        "| Measure | Value |",
        "|---|---|",
        f"| First reply valid against the schema | {sum(defined)}/{len(defined)} |",
        f"| Places the rules changed the model's text | {fixes} ({fixes / len(scores):.1f} per video) |",
        f"| Residual violations after the rules (must be 0) | {sum(violations.values())}{detail} |",
        "",
    ]


def _strata(run: LoadedRun) -> list[str]:
    stratum = {
        r.video_id: f"{(r.language or '?')[:2]} / {r.source_type}" for r in run.records if r.status == "ok"
    }
    groups: dict[str, list[VideoScore]] = {}
    for video_id, score in run.score.scores.items():
        groups.setdefault(stratum.get(video_id, "?"), []).append(score)
    lines = ["| Language / captions | Videos | Final micro-F1 |", "|---|---|---|"]
    for name, scores in sorted(groups.items()):
        lines.append(
            f"| {name} | {len(scores)} | {_pct(micro_prf([s.counts('final') for s in scores]).f1)} |"
        )
    return [*lines, "", "Strata overlap with channels and are tiny: descriptive only.", ""]


def _single(run: LoadedRun, dataset: Dataset) -> list[str]:
    scores = list(run.score.scores.values())
    lines = [f"## Run `{run.meta.run_id}`", "", *_availability(run)]
    if not scores:
        return [*lines, "No answered video to score.", ""]
    return [
        *lines,
        f"### Topics ({len(scores)} answered videos)",
        "",
        *_topic_table(scores),
        *_per_topic(scores, [t.name for t in dataset.topics]),
        "### Output format",
        "",
        *_format(scores),
        "### By stratum",
        "",
        *_strata(run),
    ]


def _drifted(run: LoadedRun, videos: Sequence[str]) -> set[str]:
    """Videos answered by another model than the one configured (OpenRouter rerouted them)."""
    answered = {r.video_id: r.answered_by for r in run.records if r.status == "ok"}
    return {v for v in videos if answered.get(v) not in (None, run.meta.model)}


def _same_inputs(a: LoadedRun, b: LoadedRun, videos: Sequence[str]) -> tuple[int, str]:
    """How many paired videos got the same input text: by hash when both runs recorded it."""
    ra = {r.video_id: r for r in a.records if r.status == "ok"}
    rb = {r.video_id: r for r in b.records if r.status == "ok"}
    if all(ra[v].input_sha256 and rb[v].input_sha256 for v in videos):
        return sum(ra[v].input_sha256 == rb[v].input_sha256 for v in videos), "text hash"
    basis = "text length, as these runs predate input hashes"
    return sum(ra[v].input_chars == rb[v].input_chars for v in videos), basis


def _comparison(a: LoadedRun, b: LoadedRun) -> list[str]:
    common = sorted(set(a.score.scores) & set(b.score.scores))
    same_config = (a.meta.model, a.meta.prompt_version) == (b.meta.model, b.meta.prompt_version)
    title = "A/A comparison (same configuration, run twice)" if same_config else "Paired comparison"
    lines = [f"## {title}: `{a.meta.run_id}` (A) vs `{b.meta.run_id}` (B)", ""]
    if not common:
        return [*lines, "No video answered by both runs.", ""]
    ca = [a.score.scores[v].counts("final") for v in common]
    cb = [b.score.scores[v].counts("final") for v in common]
    f1 = paired_bootstrap(ca, cb, f1_difference)
    precision = paired_bootstrap(ca, cb, precision_difference)
    p = paired_permutation_p(ca, cb, f1_difference)

    prf_a, prf_b = micro_prf(ca), micro_prf(cb)
    changed = sum(a.score.scores[v].topics["final"] != b.score.scores[v].topics["final"] for v in common)
    same_input, basis = _same_inputs(a, b, common)

    pairs = [(a.score.scores[v].first_try_valid, b.score.scores[v].first_try_valid) for v in common]
    only_a = sum(1 for x, y in pairs if x is True and y is False)
    only_b = sum(1 for x, y in pairs if x is False and y is True)

    drifted = _drifted(a, common) | _drifted(b, common)
    if a.meta.model == b.meta.model:  # same model: a change of upstream provider is also drift
        provider_a = {r.video_id: r.provider for r in a.records if r.status == "ok"}
        provider_b = {r.video_id: r.provider for r in b.records if r.status == "ok"}
        drifted |= {v for v in common if provider_a.get(v) != provider_b.get(v)}

    lines += [
        f"Quality is compared on the {len(common)} videos both runs answered; availability is reported "
        "per run above, so a run cannot look better by failing on hard videos.",
        "",
        f"- Inputs: identical on {same_input} of {len(common)} paired videos (compared by {basis}); "
        "both runs are scored against the same labels.",
        f"- Final topics differ on {changed} of {len(common)} videos.",
        "",
        f"| On the {len(common)} paired videos | Precision | Recall | Micro-F1 |",
        "|---|---|---|---|",
        f"| A | {_pct(prf_a.precision)} | {_pct(prf_a.recall)} | {_pct(prf_a.f1)} |",
        f"| B | {_pct(prf_b.precision)} | {_pct(prf_b.recall)} | {_pct(prf_b.f1)} |",
        "",
        "| Measure (B - A, final topics) | Difference | 95% bootstrap CI |",
        "|---|---|---|",
        f"| Micro-F1 | {_signed(f1.estimate)} | {_signed(f1.low)} to {_signed(f1.high)} |",
        f"| Precision | {_signed(precision.estimate)} | "
        f"{_signed(precision.low)} to {_signed(precision.high)} |",
        "",
        f"- Paired permutation test on the F1 difference: p = {p:.3f} "
        "(a large p means no difference was detected, not that the runs are equivalent).",
        f"- First reply valid: only A {only_a}, only B {only_b} "
        f"(exact McNemar p = {mcnemar_exact_p(only_a, only_b):.3f}).",
    ]
    if drifted:
        lines.append(
            f"- On {len(drifted)} of {len(common)} videos a different model answered than configured, "
            "or the upstream provider changed between runs; part of the difference may come from that."
        )
    if same_config:
        margin = max(abs(f1.low), abs(f1.high))
        lines += [
            "",
            f"The 95% CI of this A/A difference reaches {margin:.3f} in micro-F1. With this few videos it "
            "is a rough, provisional reference, used as the tolerated margin when comparing "
            "configurations; it will be re-estimated as runs and videos accumulate.",
        ]
    else:
        lines += [
            "",
            "Decision rules (see the evaluation plan) need a completed A/A comparison on the test set; "
            "until then this comparison is preliminary and no decision is taken from it.",
        ]
    return [*lines, ""]


def _blind(dataset: Dataset, a: LoadedRun, b: LoadedRun, session: BlindSession | None) -> list[str]:
    lines = ["### Blind human review", ""]
    if session is None or not session.judgments:
        return [
            *lines,
            "Not done: summary quality (contradictions, unsupported claims, coverage) is unmeasured.",
            "",
        ]
    items = {i.video_id: i for i in dataset.items}
    matcher = TopicMatcher(dataset.topics)
    n = len(session.judgments)
    lines += [
        f"{n} video(s) reviewed by a person who saw the outputs as X and Y in a seeded random order. "
        "Descriptive only: too few for a test.",
        "",
        "| Measure | A | B |",
        "|---|---|---|",
    ]
    rows: dict[str, list[str]] = {}
    quotes: list[str] = []
    agree = Counter[str]()
    for label, run in (("A", a), ("B", b)):
        outputs = [(j, j.for_run(run.meta.run_id)) for j in session.judgments]
        severity = Counter(i.severity for _, o in outputs for i in o.issues)
        states = Counter(s for _, o in outputs for s in o.concepts.values())
        concepts = sum(states.values())
        cells = {
            "Preferred": f"{sum(j.preferred_run() == run.meta.run_id for j, _ in outputs)} (ties "
            f"{sum(j.preferred_run() is None for j, _ in outputs)})",
            "Main takeaway conveyed": f"{sum(o.takeaway_conveyed for _, o in outputs)}/{n}",
            "Concepts explained correctly": f"{states[ConceptState.EXPLAINED]}/{concepts}",
            "Concepts explained wrongly": f"{states[ConceptState.WRONG]}/{concepts}",
            "Concepts named only / missing": (
                f"{states[ConceptState.NAME_ONLY]} / {states[ConceptState.NOT_MENTIONED]}"
            ),
            **{f"{s.value} issues": str(severity[s]) for s in Severity},
        }
        for key, value in cells.items():
            rows.setdefault(key, []).append(value)
        records = {r.video_id: r for r in run.records if r.status == "ok"}
        for judgment, output in outputs:
            quotes += [
                f"- {label}, `{judgment.video_id}`, {i.severity.value}: {i.text}"
                for i in output.issues
                if i.severity is not Severity.E4
            ]
            labels = items[judgment.video_id].labels
            if labels is None or judgment.video_id not in records:
                continue
            final = replay(records[judgment.video_id], run.titles[judgment.video_id], matcher).final
            for concept in labels.concepts:
                human = (
                    output.concepts.get(concept.term, ConceptState.NOT_MENTIONED)
                    is not ConceptState.NOT_MENTIONED
                )
                auto = concept_mentioned(final, concept)
                agree[("tp" if auto else "fn") if human else ("fp" if auto else "tn")] += 1
    lines += [f"| {key} | {values[0]} | {values[1]} |" for key, values in rows.items()]
    lines += ["", "E1-E3 issues (E4 is counted only):", "", *(quotes or ["- none"]), ""]
    checked = sum(agree.values())
    if checked:
        precision = agree["tp"] / (agree["tp"] + agree["fp"]) if agree["tp"] + agree["fp"] else None
        recall = agree["tp"] / (agree["tp"] + agree["fn"]) if agree["tp"] + agree["fn"] else None
        lines += [
            f"Automatic concept check against the person, on {checked} concept judgments (mentioned or not): "
            f"agreement {(agree['tp'] + agree['tn']) / checked:.2f}, precision {_pct(precision)}, "
            f"recall {_pct(recall)}. The automatic check stays an alarm-only metric until this "
            "agreement holds up.",
            "",
        ]
    return lines


def build_report(dataset: Dataset, runs: Sequence[LoadedRun], *, blind: BlindSession | None = None) -> str:
    lines = [
        "# Evaluation report",
        "",
        "Observed under the conditions below. With this few videos, differences that are not "
        "significant are not evidence of no difference.",
        "",
        *_conditions(dataset, runs),
    ]
    for run in runs:
        lines += _single(run, dataset)
    if len(runs) == 2:
        lines += _comparison(runs[0], runs[1])
        if (
            runs[0].meta.model != runs[1].meta.model
            or runs[0].meta.prompt_version != runs[1].meta.prompt_version
        ):
            lines += _blind(dataset, runs[0], runs[1], blind)
    return "\n".join(lines).rstrip() + "\n"
