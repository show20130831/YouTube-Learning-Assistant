from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest
import requests
from sqlalchemy import Engine, select, text
from sqlalchemy.orm import Session
from typer.testing import CliRunner
from youtube_transcript_api import (
    IpBlocked,
    PoTokenRequired,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeRequestFailed,
)

from yla import cli
from yla.config import AppSettings
from yla.db.models import Channel, FetchedBy, SourceType, Transcript, User, Video, VideoStatus
from yla.sync import sync_config
from yla.worker.local_worker import fetch_transcripts, select_transcript_candidates
from yla.youtube.transcripts import (
    DatabaseTranscriptProvider,
    DatabaseTranscriptSink,
    TranscriptResult,
    TranscriptStatus,
    YouTubeTranscriptProvider,
    _TimeoutSession,
)

# --- fakes for youtube-transcript-api ---------------------------------------------------


@dataclass
class Snippet:
    text: str


@dataclass
class Track:
    language_code: str
    is_generated: bool
    lines: list[str] = field(default_factory=lambda: ["hello", "world"])

    def fetch(self) -> list[Snippet]:
        return [Snippet(t) for t in self.lines]


class FakeApi:
    """Returns (or raises) one scripted outcome per call; the last one repeats."""

    def __init__(self, *outcomes: list[Track] | Exception) -> None:
        self.outcomes = list(outcomes)
        self.calls = 0

    def list(self, video_id: str) -> list[Track]:
        outcome = self.outcomes[min(self.calls, len(self.outcomes) - 1)]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def provider(
    *outcomes: list[Track] | Exception, sleeps: list[float] | None = None
) -> YouTubeTranscriptProvider:
    sink = sleeps if sleeps is not None else []
    return YouTubeTranscriptProvider(api=FakeApi(*outcomes), sleep=sink.append)


def test_manual_track_preferred_over_generated() -> None:
    result = provider([Track("en", True, ["auto"]), Track("en", False, ["manual"])]).fetch("v")
    assert (result.status, result.text) == (TranscriptStatus.MANUAL, "manual")


def test_language_preference_within_same_kind() -> None:
    result = provider([Track("ja", True), Track("zh-TW", True), Track("en", True, ["english"])]).fetch("v")
    assert (result.status, result.language, result.text) == (TranscriptStatus.GENERATED, "en", "english")


def test_snippets_joined_one_per_line() -> None:
    assert provider([Track("en", False, ["a", "b"])]).fetch("v").text == "a\nb"


@pytest.mark.parametrize("tracks", [[], [Track("en", True, ["", "  "])]])
def test_no_tracks_or_empty_text_is_none(tracks: list[Track]) -> None:
    assert provider(tracks).fetch("v").status is TranscriptStatus.NONE


def _http_failed(status: int) -> YouTubeRequestFailed:
    return YouTubeRequestFailed(
        "v",
        requests.HTTPError(f"{status} Client Error: Some Reason for url: https://yt/timedtext?signature=s"),
    )


@pytest.mark.parametrize(
    ("exc", "status"),
    [
        (RequestBlocked("v"), TranscriptStatus.BLOCKED),
        (IpBlocked("v"), TranscriptStatus.BLOCKED),
        (_http_failed(429), TranscriptStatus.BLOCKED),
        (_http_failed(500), TranscriptStatus.ERROR),
        (TranscriptsDisabled("v"), TranscriptStatus.NONE),
        (VideoUnavailable("v"), TranscriptStatus.NONE),
        (PoTokenRequired("v"), TranscriptStatus.ERROR),
        (requests.ConnectionError("offline"), TranscriptStatus.ERROR),
    ],
)
def test_errors_are_classified(exc: Exception, status: TranscriptStatus) -> None:
    result = provider(exc).fetch("v")
    assert result.status is status
    assert result.error and type(exc).__name__ in result.error


def test_transient_server_error_is_retried_then_succeeds() -> None:
    sleeps: list[float] = []
    result = provider(_http_failed(502), [Track("en", True)], sleeps=sleeps).fetch("v")
    assert result.status is TranscriptStatus.GENERATED
    assert sleeps == [5.0]


def test_persistent_server_error_gives_readable_error_without_url() -> None:
    api = FakeApi(_http_failed(502))
    sleeps: list[float] = []
    result = YouTubeTranscriptProvider(api=api, sleep=sleeps.append).fetch("v")
    assert result.status is TranscriptStatus.ERROR
    assert result.error == "YouTubeRequestFailed: 502 Client Error: Some Reason"
    assert (api.calls, sleeps) == (3, [5.0, 10.0])


@pytest.mark.parametrize("exc", [_http_failed(429), RequestBlocked("v"), TranscriptsDisabled("v")])
def test_blocks_and_missing_captions_are_not_retried(exc: Exception) -> None:
    api = FakeApi(exc)
    YouTubeTranscriptProvider(api=api, sleep=lambda _: None).fetch("v")
    assert api.calls == 1


def test_timeouts_are_retried() -> None:
    result = provider(requests.Timeout("slow"), [Track("en", False)]).fetch("v")
    assert result.status is TranscriptStatus.MANUAL


def test_http_session_applies_default_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_request(self: requests.Session, *args: object, **kwargs: object) -> str:
        seen.update(kwargs)
        return "ok"

    monkeypatch.setattr(requests.Session, "request", fake_request)
    session = _TimeoutSession(12.0)
    session.request("GET", "https://example.com")
    session.request("GET", "https://example.com", timeout=3)
    assert seen["timeout"] == 3
    _TimeoutSession(12.0).request("GET", "https://example.com")
    assert seen["timeout"] == 12.0


# --- database sink / provider and the local worker -------------------------------------

NOW = datetime(2026, 10, 6, 2, 30, tzinfo=UTC)
LC = "UCC-lyoTfSrcJzA1ab3APAgw"


def user_id(session: Session) -> int:
    return session.scalars(select(User.id)).one()


def add_videos(session: Session, specs: list[tuple[str, VideoStatus, int]]) -> dict[str, Video]:
    """specs: (youtube id, status, discovered days ago); channel LangChain is subscribed."""
    sync_config(
        session, AppSettings.model_validate({"channels": [{"handle": "LangChain", "channel_id": LC}]})
    )
    channel_id = session.scalars(select(Channel.id)).one()
    videos = {}
    for i, (vid, status, days_ago) in enumerate(specs):
        when = NOW - timedelta(days=days_ago)
        videos[vid] = Video(
            youtube_video_id=vid,
            channel_id=channel_id,
            title=vid,
            published_at=when - timedelta(minutes=i),
            discovered_at=when,
            status=status,
        )
    session.add_all(videos.values())
    session.flush()
    return videos


class ScriptedProvider:
    def __init__(self, results: dict[str, TranscriptResult]) -> None:
        self.results = results
        self.calls: list[str] = []

    def fetch(self, video_id: str) -> TranscriptResult:
        self.calls.append(video_id)
        return self.results.get(video_id, TranscriptResult(status=TranscriptStatus.NONE))


GOOD = TranscriptResult(status=TranscriptStatus.GENERATED, language="en", text="some words")


@pytest.mark.db
def test_sink_and_database_provider_round_trip(session: Session) -> None:
    video = add_videos(session, [("v1", VideoStatus.PENDING, 0)])["v1"]
    sink = DatabaseTranscriptSink(session)
    sink.save(video.id, TranscriptResult(status=TranscriptStatus.NONE))  # nothing stored
    assert DatabaseTranscriptProvider(session).get(video.id).status is TranscriptStatus.NONE

    sink.save(video.id, GOOD)
    row = session.get(Transcript, video.id)
    assert row is not None
    assert (row.source_type, row.char_count, row.fetched_by) == (SourceType.AUTO_CAPTION, 10, FetchedBy.LOCAL)
    assert DatabaseTranscriptProvider(session).get(video.id) == GOOD

    sink.save(video.id, TranscriptResult(status=TranscriptStatus.MANUAL, language="en", text="better"))
    assert DatabaseTranscriptProvider(session).get(video.id).status is TranscriptStatus.MANUAL


@pytest.mark.db
def test_candidates_are_recent_unfinished_videos_without_transcripts(session: Session) -> None:
    videos = add_videos(
        session,
        [
            ("pending", VideoStatus.PENDING, 0),
            ("deferred", VideoStatus.DEFERRED, 1),
            ("star1", VideoStatus.ANALYZED, 2),  # analysed without captions -> may be upgraded
            ("old", VideoStatus.PENDING, 4),  # outside the 3-day window
            ("backfill", VideoStatus.SKIPPED_BACKFILL, 0),
            ("short", VideoStatus.SKIPPED_SHORT, 0),
            ("has_text", VideoStatus.PENDING, 0),
        ],
    )
    DatabaseTranscriptSink(session).save(videos["has_text"].id, GOOD)

    found = select_transcript_candidates(session, user_id(session), now=NOW, lookback_days=3, limit=10)
    assert [v.youtube_video_id for v in found] == ["pending", "deferred", "star1"]


@pytest.mark.db
def test_worker_saves_reports_and_paces_requests(session: Session) -> None:
    add_videos(
        session, [("a", VideoStatus.PENDING, 0), ("b", VideoStatus.PENDING, 0), ("c", VideoStatus.PENDING, 0)]
    )
    scripted = ScriptedProvider(
        {"a": GOOD, "c": TranscriptResult(status=TranscriptStatus.ERROR, error="PoTokenRequired: x")}
    )
    sleeps: list[float] = []

    report = fetch_transcripts(
        session,
        user_id(session),
        scripted,
        DatabaseTranscriptSink(session),
        now=NOW,
        limit=10,
        sleep=sleeps.append,
    )
    assert report.attempted == 3
    assert report.saved == {"a": TranscriptStatus.GENERATED}
    assert report.no_captions == ["b"]
    assert report.errors == {"c": "PoTokenRequired: x"}
    assert sleeps == [1.5, 1.5]  # between requests, not before the first


@pytest.mark.db
def test_worker_stops_at_first_block(session: Session) -> None:
    add_videos(session, [("a", VideoStatus.PENDING, 0), ("b", VideoStatus.PENDING, 0)])
    scripted = ScriptedProvider(
        {"a": TranscriptResult(status=TranscriptStatus.BLOCKED, error="RequestBlocked")}
    )

    report = fetch_transcripts(
        session, user_id(session), scripted, DatabaseTranscriptSink(session), now=NOW, limit=10, sleep=print
    )
    assert report.blocked
    assert scripted.calls == ["a"]


@pytest.mark.db
def test_worker_respects_limit(session: Session) -> None:
    add_videos(session, [(f"v{i}", VideoStatus.PENDING, 0) for i in range(5)])
    scripted = ScriptedProvider({})
    fetch_transcripts(
        session,
        user_id(session),
        scripted,
        DatabaseTranscriptSink(session),
        now=NOW,
        limit=2,
        sleep=lambda _: None,
    )
    assert scripted.calls == ["v0", "v1"]  # newest first


# --- CLI ---------------------------------------------------------------------------------


@pytest.fixture
def cli_db(engine, test_database_url, tmp_path, monkeypatch):  # type: ignore[no-untyped-def]
    """CLI commands commit, so they run against the test DB and are cleaned up afterwards."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", test_database_url)
    yield engine
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE users, channels, job_runs RESTART IDENTITY CASCADE"))


def use_provider(monkeypatch: pytest.MonkeyPatch, results: dict[str, TranscriptResult]) -> ScriptedProvider:
    scripted = ScriptedProvider(results)
    monkeypatch.setattr(cli, "YouTubeTranscriptProvider", lambda: scripted)
    return scripted


def seed_pending(engine: Engine, ids: list[str]) -> None:
    with Session(engine) as db:
        add_videos(db, [(vid, VideoStatus.PENDING, 0) for vid in ids])
        db.query(Video).update({Video.discovered_at: datetime.now(UTC)})
        db.commit()


@pytest.mark.db
@pytest.mark.parametrize(
    ("status", "exit_code"),
    [(TranscriptStatus.GENERATED, 0), (TranscriptStatus.NONE, 0), (TranscriptStatus.BLOCKED, 2)],
)
def test_cli_worker_check(
    cli_db: Engine, monkeypatch: pytest.MonkeyPatch, status: TranscriptStatus, exit_code: int
) -> None:
    use_provider(monkeypatch, {"vidX": TranscriptResult(status=status, text="x", error="RequestBlocked")})
    result = CliRunner().invoke(cli.app, ["worker", "--check", "--video-id", "vidX"])
    assert result.exit_code == exit_code, result.output
    assert f"Video vidX: {status.value}" in result.output


@pytest.mark.db
def test_cli_worker_check_needs_a_video(cli_db: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    use_provider(monkeypatch, {})
    assert CliRunner().invoke(cli.app, ["worker", "--check"]).exit_code == 1


@pytest.mark.db
def test_cli_worker_fetches_and_stores(cli_db: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_pending(cli_db, ["a", "b"])
    use_provider(monkeypatch, {"a": GOOD})
    monkeypatch.setattr("yla.worker.local_worker.time.sleep", lambda _: None)

    result = CliRunner().invoke(cli.app, ["worker", "--skip-discover"])
    assert result.exit_code == 0, result.output
    assert "Fetched captions for 1 of 2 videos" in result.output
    with Session(cli_db) as db:
        assert len(db.scalars(select(Transcript)).all()) == 1


@pytest.mark.db
def test_cli_worker_exits_2_when_blocked(cli_db: Engine, monkeypatch: pytest.MonkeyPatch) -> None:
    seed_pending(cli_db, ["a"])
    use_provider(monkeypatch, {"a": TranscriptResult(status=TranscriptStatus.BLOCKED, error="IpBlocked")})
    result = CliRunner().invoke(cli.app, ["worker", "--skip-discover"])
    assert result.exit_code == 2
    assert "blocking caption requests" in result.output


# --- run_local_worker (job record) -------------------------------------------------------


@pytest.mark.db
def test_run_local_worker_records_a_job(session: Session) -> None:
    from yla.db.models import JobRun, JobStatus
    from yla.worker.local_worker import run_local_worker

    add_videos(session, [("a", VideoStatus.PENDING, 0), ("b", VideoStatus.PENDING, 0)])
    user = session.scalars(select(User)).one()
    report = run_local_worker(
        session,
        user,
        ScriptedProvider({"a": GOOD}),
        DatabaseTranscriptSink(session),
        now=NOW,
        sleep=lambda _: None,
    )

    assert report.discovery is None and report.transcripts.saved == {"a": TranscriptStatus.GENERATED}
    job = session.scalars(select(JobRun).where(JobRun.job_name == "local_worker")).one()
    assert job.status is JobStatus.SUCCEEDED
    assert {k: job.stats[k] for k in ("saved", "attempted", "auto", "no_captions", "blocked")} == {
        "saved": 1,
        "attempted": 2,
        "auto": 1,
        "no_captions": 1,
        "blocked": False,
    }
    assert job.stats["host"]


@pytest.mark.db
def test_run_local_worker_records_failures(session: Session) -> None:
    from yla.db.models import JobRun, JobStatus
    from yla.worker.local_worker import run_local_worker

    add_videos(session, [("a", VideoStatus.PENDING, 0)])
    user = session.scalars(select(User)).one()

    class Broken:
        def fetch(self, video_id: str) -> TranscriptResult:
            raise RuntimeError("disk full")

    with pytest.raises(RuntimeError):
        run_local_worker(session, user, Broken(), DatabaseTranscriptSink(session), now=NOW)
    job = session.scalars(select(JobRun).where(JobRun.job_name == "local_worker")).one()
    assert (job.status, job.error) == (JobStatus.FAILED, "RuntimeError: disk full")


@pytest.mark.db
def test_cli_status_lists_recent_jobs(cli_db: Engine) -> None:
    from yla.db.models import JobRun, JobStatus

    with Session(cli_db) as db:
        add_videos(db, [])
        db.add_all(
            [
                JobRun(
                    job_name="local_worker",
                    status=JobStatus.SUCCEEDED,
                    stats={"host": "pc", "saved": 2, "attempted": 3, "blocked": True},
                ),
                JobRun(
                    job_name="daily_pipeline",
                    status=JobStatus.SUCCEEDED,
                    stats={"analyzed": 4, "llm_calls": 4, "quota_exhausted": True},
                ),
                JobRun(
                    job_name="send_digest",
                    status=JobStatus.FAILED,
                    stats={"messages": 3},
                    error="HTTP 401: bad token",
                ),
            ]
        )
        db.commit()
    result = CliRunner().invoke(cli.app, ["status"])
    assert result.exit_code == 0, result.output
    assert "[pc] new -, captions 2/3, BLOCKED" in result.output
    assert "analysed 4 (4 LLM)" in result.output and "QUOTA USED UP" in result.output
    assert "HTTP 401: bad token" in result.output


@pytest.mark.db
def test_cli_status_with_no_runs(cli_db: Engine) -> None:
    with Session(cli_db) as db:
        add_videos(db, [])
        db.commit()
    assert "No job runs" in CliRunner().invoke(cli.app, ["status"]).output
