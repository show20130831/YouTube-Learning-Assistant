"""Run one model configuration over the dataset and record every answer.

Kept apart from production on purpose: nothing here touches the database. Writing analyses
there would put evaluation videos into the next LINE digest, use up the daily analysis limit
and inflate the trending topics.

- One model per run, with fallback off: otherwise a run would mix answers from several models.
- Raw outputs are stored, so scoring can replay any version of the post-processing rules
  without spending LLM requests.
- Resumable: videos that already have an answer are skipped; failed ones are retried on the
  next invocation (the free quota allows about one configuration per day). A 429 stops the
  run, since the daily quota is most likely used up.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from yla.analysis.analyzer import RULES_VERSION, AnalysisFailed, Analyzer
from yla.content.tiering import choose_content
from yla.eval.dataset import DEFAULT_RUNS_DIR, Cache, Dataset
from yla.llm.client import ChatMessage, LLMClient, LLMResponse
from yla.llm.prompts import PROMPT_VERSION


class RunMeta(BaseModel):
    run_id: str
    model: str
    prompt_version: str
    rules_version: str  # rules in force when the run was made; scoring may replay newer ones
    dataset_version: str  # when the run started; it may continue on later versions
    temperature: float | None
    top_p: float | None = None  # not sent: the provider default applies
    created_at: datetime


class Record(BaseModel):
    """One attempt at one video. A video can have several (a failure, then a retry)."""

    video_id: str
    attempt_round: int
    started_at: datetime
    status: str  # "ok" | "timeout" | "rate_limited" | "invalid" | "error" | "no_text"
    latency_seconds: float
    calls: int
    attempts: list[tuple[str, str]]  # per request: (model, result)
    answered_by: str | None = None
    provider: str | None = None
    usage: dict[str, Any] = {}
    source_type: str | None = None
    language: str | None = None
    input_chars: int | None = None
    input_sha256: str | None = None  # of the text sent, so paired runs can prove identical inputs
    raw: dict[str, Any] | None = None
    error: str | None = None


class RecordingClient:
    """Wraps the LLM client to keep the provider and token usage of the last answer."""

    def __init__(self, inner: LLMClient) -> None:
        self._inner = inner
        self.last: LLMResponse | None = None

    def complete_json(
        self, messages: list[ChatMessage], *, schema: dict[str, Any], schema_name: str, model: str
    ) -> LLMResponse:
        self.last = None
        response = self._inner.complete_json(messages, schema=schema, schema_name=schema_name, model=model)
        self.last = response
        return response


@dataclass
class RunSummary:
    attempted: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0
    stopped_on_rate_limit: bool = False


class Run:
    def __init__(self, root: Path) -> None:
        self.root = root

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def records_path(self) -> Path:
        return self.root / "records.jsonl"

    def meta(self) -> RunMeta:
        return RunMeta.model_validate_json(self.meta_path.read_text(encoding="utf-8"))

    def records(self) -> list[Record]:
        if not self.records_path.exists():
            return []
        lines = self.records_path.read_text(encoding="utf-8").splitlines()
        return [Record.model_validate_json(line) for line in lines if line.strip()]

    def append(self, record: Record) -> None:
        with self.records_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump(mode="json"), ensure_ascii=False) + "\n")


def open_run(run_id: str, runs_dir: Path = DEFAULT_RUNS_DIR) -> Run:
    run = Run(runs_dir / run_id)
    if not run.meta_path.exists():
        raise FileNotFoundError(f"no evaluation run {run_id} in {runs_dir}")
    return run


def create_run(meta: RunMeta, runs_dir: Path = DEFAULT_RUNS_DIR) -> Run:
    run = Run(runs_dir / meta.run_id)
    if run.meta_path.exists():
        existing = run.meta()
        # A run may continue on a newer dataset version: versions only add videos, and labels of
        # videos already used are frozen. The model and prompt must not change within a run.
        if (existing.model, existing.prompt_version) != (meta.model, meta.prompt_version):
            raise ValueError(f"run {meta.run_id} exists with a different model or prompt")
        return run
    run.root.mkdir(parents=True, exist_ok=True)
    run.meta_path.write_text(meta.model_dump_json(indent=1), encoding="utf-8")
    return run


def new_meta(
    *, run_id: str, model: str, dataset: Dataset, temperature: float | None, now: datetime | None = None
) -> RunMeta:
    return RunMeta(
        run_id=run_id,
        model=model,
        prompt_version=PROMPT_VERSION,
        rules_version=RULES_VERSION,
        dataset_version=dataset.version,
        temperature=temperature,
        created_at=now or datetime.now(UTC),
    )


def _result_of(attempts: list[tuple[str, str]]) -> str:
    return attempts[-1][1] if attempts else "error"


def run_dataset(
    run: Run,
    dataset: Dataset,
    cache: Cache,
    client: LLMClient,
    *,
    limit: int | None = None,
    clock: Callable[[], float] = time.monotonic,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
    on_record: Callable[[Record], None] | None = None,
) -> RunSummary:
    meta = run.meta()
    recording = RecordingClient(client)
    analyzer = Analyzer(recording, [meta.model], dataset.topics)  # one model: fallback off
    previous: dict[str, list[Record]] = {}
    for record in run.records():
        previous.setdefault(record.video_id, []).append(record)

    summary = RunSummary()
    for item in dataset.labelled:
        done = previous.get(item.video_id, [])
        if any(r.status in ("ok", "no_text") for r in done):
            summary.skipped += 1
            continue
        video = cache.get(item.video_id)
        if video is None:
            summary.skipped += 1  # transcript not fetched yet
            continue
        if limit is not None and summary.attempted >= limit:
            break

        tier = choose_content(
            video.transcript(),
            video.title,
            video.description,
            min_description_chars=dataset.min_description_chars,
        )
        base = {
            "video_id": item.video_id,
            "attempt_round": len(done) + 1,
            "started_at": now(),
            "source_type": tier.source_type.value,
            "language": tier.language,
            "input_chars": len(tier.text),
            "input_sha256": hashlib.sha256(tier.text.encode()).hexdigest()[:16],
        }
        summary.attempted += 1
        if not tier.available:
            record = Record(**base, status="no_text", latency_seconds=0, calls=0, attempts=[])
        else:
            start = clock()
            try:
                outcome = analyzer.analyze(title=video.title, channel=video.channel, tier=tier)
            except AnalysisFailed as exc:
                record = Record(
                    **base,
                    status=_result_of(exc.attempts),
                    latency_seconds=round(clock() - start, 2),
                    calls=exc.calls,
                    attempts=exc.attempts,
                    error=str(exc)[:500],
                )
            else:
                last = recording.last
                record = Record(
                    **base,
                    status="ok",
                    latency_seconds=round(clock() - start, 2),
                    calls=outcome.calls,
                    attempts=outcome.attempts,
                    answered_by=outcome.model,
                    provider=last.provider if last else None,
                    usage=last.usage if last else {},
                    raw=outcome.raw,
                )
        run.append(record)
        if on_record:
            on_record(record)
        if record.status == "ok":
            summary.ok += 1
        elif record.status != "no_text":
            summary.failed += 1
        if record.status == "rate_limited":
            summary.stopped_on_rate_limit = True
            break
    return summary
