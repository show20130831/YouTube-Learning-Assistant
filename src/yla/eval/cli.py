"""``yla eval ...``: fill the transcript cache, run a model over the dataset, write a report."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated
from zoneinfo import ZoneInfo

import httpx
import typer

from yla.analysis.topics import TopicMatcher
from yla.eval.blind import (
    BlindSession,
    ConceptState,
    Issue,
    OutputJudgment,
    PlannedPair,
    Preference,
    Severity,
    load_session,
    new_judgment,
    plan_pairs,
    record_judgment,
    render_output,
    save_session,
)
from yla.eval.dataset import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATASET_PATH,
    DEFAULT_RUNS_DIR,
    Cache,
    Dataset,
    Labels,
    load_dataset,
)
from yla.eval.fetch import VideoInfo, fill_cache
from yla.eval.report import build_report, load_run
from yla.eval.runner import Record, create_run, new_meta, open_run, run_dataset

eval_app = typer.Typer(help="Evaluate models, prompts and rules on a labelled dataset.", no_args_is_help=True)

DatasetPath = Annotated[Path, typer.Option("--dataset", help="Dataset YAML")]
CacheDir = Annotated[Path, typer.Option("--cache", help="Transcript cache folder (git-ignored)")]
RunsDir = Annotated[Path, typer.Option("--runs", help="Run output folder (git-ignored)")]
TAIPEI = ZoneInfo("Asia/Taipei")


def _fail(message: str) -> typer.Exit:
    typer.secho(message, fg=typer.colors.RED, err=True)
    return typer.Exit(code=1)


@eval_app.command("fetch")
def fetch_command(
    dataset_path: DatasetPath = DEFAULT_DATASET_PATH,
    cache_dir: CacheDir = DEFAULT_CACHE_DIR,
    export_only: Annotated[
        bool, typer.Option(help="Only export transcripts already in the database")
    ] = False,
) -> None:
    """Cache titles and transcripts: from the database (read-only), else from YouTube (max 5 a day)."""
    from sqlalchemy import select

    from yla.config import Secrets
    from yla.db.models import Channel, Transcript, Video
    from yla.db.session import make_engine, session_scope
    from yla.youtube.transcripts import DatabaseTranscriptProvider, YouTubeTranscriptProvider

    dataset = load_dataset(dataset_path)
    ids = [i.video_id for i in dataset.items]
    database_url = Secrets().database_url
    if database_url is None:
        raise _fail("DATABASE_URL is not set: titles and descriptions come from the database")
    engine = make_engine(database_url.get_secret_value())
    with session_scope(engine) as session:
        rows = session.execute(
            select(Video, Channel)
            .join(Channel, Channel.id == Video.channel_id)
            .where(Video.youtube_video_id.in_(ids))
        ).all()
        info = {
            v.youtube_video_id: VideoInfo(c.title or c.handle, v.title, v.description or "") for v, c in rows
        }
        with_text = set(session.scalars(select(Transcript.video_id)))
        reader = DatabaseTranscriptProvider(session)
        stored = {v.youtube_video_id: reader.get(v.id) for v, _ in rows if v.id in with_text}

    report = fill_cache(
        dataset,
        Cache(cache_dir),
        info=info,
        stored=stored,
        provider=None if export_only else YouTubeTranscriptProvider(),
        now=lambda: datetime.now(UTC),
        tz=TAIPEI,
    )
    typer.echo(f"Exported from the database: {len(report.exported)}")
    for video_id, status in report.fetched.items():
        typer.echo(f"Fetched {video_id}: {status}")
    if report.left_for_later:
        typer.secho(
            f"Left for another day (daily limit): {', '.join(report.left_for_later)}", fg=typer.colors.YELLOW
        )
    if report.missing_info:
        typer.secho(f"Not in the database: {', '.join(report.missing_info)}", fg=typer.colors.YELLOW)
    if report.blocked:
        raise _fail("YouTube blocked the caption request: stopped (see docs/operations.md)")


def _default_run_id(model: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model.split("/")[-1].removesuffix(":free").lower()).strip("-")
    return f"{datetime.now(TAIPEI):%Y%m%d}-{slug}"


@eval_app.command("run")
def run_command(
    model: Annotated[str, typer.Option(help="OpenRouter model ID; one model, no fallback")],
    run_id: Annotated[str | None, typer.Option(help="Default: <date>-<model>")] = None,
    limit: Annotated[int | None, typer.Option(help="Attempt at most this many videos")] = None,
    input_mode: Annotated[
        str,
        typer.Option(
            "--input", help="transcript, or title: the title_only_subset from title and description only"
        ),
    ] = "transcript",
    dataset_path: DatasetPath = DEFAULT_DATASET_PATH,
    cache_dir: CacheDir = DEFAULT_CACHE_DIR,
    runs_dir: RunsDir = DEFAULT_RUNS_DIR,
) -> None:
    """Analyse every labelled, cached video with one model. Never writes to the database."""
    from yla.config import Secrets
    from yla.llm.client import OpenRouterClient

    key = Secrets().openrouter_api_key
    if key is None:
        raise _fail("OPENROUTER_API_KEY is not set")
    dataset = load_dataset(dataset_path)
    with httpx.Client(timeout=httpx.Timeout(30.0, read=200.0)) as http:
        client = OpenRouterClient(key.get_secret_value(), http)
        meta = new_meta(
            run_id=run_id or _default_run_id(model) + ("-title" if input_mode == "title" else ""),
            model=model,
            dataset=dataset,
            temperature=client.temperature,
            input=input_mode,
        )
        run = create_run(meta, runs_dir)

        def show(record: Record) -> None:
            typer.echo(
                f"{record.video_id}  {record.status:<12} "
                f"{record.latency_seconds:6.1f}s  {record.calls} call(s)"
            )

        summary = run_dataset(run, dataset, Cache(cache_dir), client, limit=limit, on_record=show)
    typer.echo(
        f"Run {meta.run_id}: {summary.ok} answered, {summary.failed} failed, {summary.skipped} skipped"
        " (already answered or not cached)."
    )
    if summary.stopped_on_rate_limit:
        typer.secho(
            "Stopped on a rate limit: the free quota is probably used up. Resume tomorrow.",
            fg=typer.colors.YELLOW,
        )


@eval_app.command("report")
def report_command(
    run_ids: Annotated[list[str], typer.Argument(help="One run, or two to compare (A then B)")],
    output: Annotated[
        Path | None, typer.Option(help="Write the Markdown here instead of printing it")
    ] = None,
    no_blind: Annotated[
        bool, typer.Option(help="Compare two configurations without the blind human review")
    ] = False,
    dataset_path: DatasetPath = DEFAULT_DATASET_PATH,
    cache_dir: CacheDir = DEFAULT_CACHE_DIR,
    runs_dir: RunsDir = DEFAULT_RUNS_DIR,
) -> None:
    """Score runs from their raw outputs with the current rules, and compare two runs.

    Comparing two configurations needs the blind review first (``yla eval blind``): seeing the
    scores before judging the summaries would bias the judgments.
    """
    if not 1 <= len(run_ids) <= 2:
        raise _fail("give one run, or two to compare")
    dataset = load_dataset(dataset_path)
    titles = {v.video_id: v.title for v in Cache(cache_dir).all()}
    try:
        runs = [load_run(open_run(r, runs_dir), dataset, titles) for r in run_ids]
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc
    session = None
    if len(runs) == 2 and (runs[0].meta.model, runs[0].meta.prompt_version) != (
        runs[1].meta.model,
        runs[1].meta.prompt_version,
    ):
        session = load_session(runs_dir, run_ids[0], run_ids[1])
        planned = plan_pairs(dataset, run_ids[0], runs[0].records, run_ids[1], runs[1].records)
        missing = {p.video_id for p in planned} - (session.judged() if session else set())
        if missing and not no_blind:
            raise _fail(
                f"The blind review is not finished ({len(missing)} of {len(planned)} videos left): run "
                f"`yla eval blind {run_ids[0]} {run_ids[1]}` first, or pass --no-blind to report without it."
            )
    text = build_report(dataset, runs, blind=session)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        typer.echo(f"Report written to {output}")
    else:
        typer.echo(text)


def _ask(prompt: str, choices: dict[str, str]) -> str:
    keys = "/".join(choices)
    while True:
        answer = typer.prompt(f"{prompt} [{keys}]").strip().lower()
        if answer in choices:
            return choices[answer]
        typer.echo(f"Answer one of: {keys}")


CONCEPT_KEYS = {
    "n": ConceptState.NOT_MENTIONED,
    "m": ConceptState.NAME_ONLY,
    "c": ConceptState.EXPLAINED,
    "w": ConceptState.WRONG,
}


def _judge_output(name: str, labels: Labels) -> OutputJudgment:
    typer.secho(f"Judging output {name}", bold=True)
    concepts = {
        c.term: ConceptState(
            _ask(f"  {c.term}: n=not mentioned, m=named only, c=explained, w=wrong", dict(CONCEPT_KEYS))
        )
        for c in labels.concepts
    }
    takeaway = _ask("  Main takeaway conveyed?", {"y": "y", "n": "n"}) == "y"
    issues = []
    typer.echo("  Problems: '<E1|E2|E3|E4> <quoted text>' per line, empty line to finish.")
    while line := typer.prompt("  ", default="", show_default=False).strip():
        level, _, text = line.partition(" ")
        if level.upper() not in Severity.__members__ or not text.strip():
            typer.echo("  Start with E1, E2, E3 or E4, then the quoted text.")
            continue
        issues.append(Issue(severity=Severity(level.upper()), text=text.strip()))
    return OutputJudgment(concepts=concepts, takeaway_conveyed=takeaway, issues=issues)


def _show_pair(dataset: Dataset, pair: PlannedPair, outputs: dict[str, str], title: str) -> Labels:
    item = next(i for i in dataset.items if i.video_id == pair.video_id)
    if item.labels is None or not item.labels.concepts or not item.labels.main_takeaway:
        raise _fail(f"{pair.video_id} needs concept and main-takeaway labels before the blind review")
    labels = item.labels
    typer.echo()
    typer.secho(f"=== {pair.video_id}: {title}", bold=True)
    typer.echo(f"Main takeaway: {labels.main_takeaway}")
    typer.echo(f"  evidence: {labels.takeaway_evidence}")
    typer.echo("Concepts: " + ", ".join(c.term for c in labels.concepts))
    for name in ("X", "Y"):
        typer.echo()
        typer.secho(f"--- Output {name}", bold=True)
        typer.echo(outputs[name])
    return labels


@eval_app.command("blind")
def blind_command(
    run_a: Annotated[str, typer.Argument(help="Current configuration")],
    run_b: Annotated[str, typer.Argument(help="Candidate configuration")],
    dataset_path: DatasetPath = DEFAULT_DATASET_PATH,
    cache_dir: CacheDir = DEFAULT_CACHE_DIR,
    runs_dir: RunsDir = DEFAULT_RUNS_DIR,
) -> None:
    """Review the blind subset without knowing which run wrote which output. Resumable."""
    from yla.eval.scoring import replay

    dataset = load_dataset(dataset_path)
    titles = {v.video_id: v.title for v in Cache(cache_dir).all()}
    try:
        records = {r: open_run(r, runs_dir).records() for r in (run_a, run_b)}
    except FileNotFoundError as exc:
        raise _fail(str(exc)) from exc
    session = load_session(runs_dir, run_a, run_b) or BlindSession(run_a=run_a, run_b=run_b)
    matcher = TopicMatcher(dataset.topics)
    pairs = [
        p
        for p in plan_pairs(dataset, run_a, records[run_a], run_b, records[run_b], seed=session.seed)
        if p.video_id not in session.judged()
    ]
    typer.echo(f"{len(session.judged())} reviewed, {len(pairs)} to go.")
    for pair in pairs:
        outputs = {}
        for name, run in (("X", pair.x_run), ("Y", pair.y_run)):
            record = next(r for r in records[run] if r.video_id == pair.video_id and r.status == "ok")
            outputs[name] = render_output(replay(record, titles[pair.video_id], matcher).final)
        labels = _show_pair(dataset, pair, outputs, titles[pair.video_id])
        x, y = _judge_output("X", labels), _judge_output("Y", labels)
        preference = Preference(_ask("Which is better?", {"x": "X", "y": "Y", "t": "tie"}))
        session = record_judgment(session, new_judgment(pair, x, y, preference))
        save_session(runs_dir, session)
        typer.echo("Saved.")
    typer.echo("Blind review complete.")
