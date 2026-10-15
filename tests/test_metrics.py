from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from yla.config import AppSettings
from yla.db.models import (
    Analysis,
    Channel,
    JobRun,
    JobStatus,
    LLMRequest,
    LLMResult,
    SourceType,
    User,
    UserVideoRelevance,
    Video,
    VideoStatus,
)
from yla.ops.alerts import trend_warnings
from yla.ops.metrics import collect_metrics, percentile, render_metrics, same_model
from yla.sync import sync_config

NOW = datetime.now(UTC)
TZ = ZoneInfo("Asia/Taipei")


def test_same_model_ignores_free_suffix_and_routers() -> None:
    assert same_model("vendor/m:free", "vendor/m")
    assert not same_model("vendor/m:free", "vendor/other:free")
    assert same_model("openrouter/free", "vendor/anything")  # routers pick a model by design
    assert same_model("vendor/m:free", None)


def test_percentile_is_nearest_rank() -> None:
    assert percentile([], 0.5) is None
    assert percentile([30, 10, 20], 0.5) == 20
    assert percentile(list(range(1, 20)), 0.95) == 19  # with few values p95 is the slowest


def setup(session: Session) -> tuple[User, int, list[Video]]:
    sync_config(
        session,
        AppSettings.model_validate(
            {"channels": [{"handle": "LangChain"}], "topics": [{"name": "AI Agent"}, {"name": "RAG"}]}
        ),
    )
    user = session.scalars(select(User)).one()
    channel_id = session.scalars(select(Channel.id)).one()
    job = JobRun(job_name="daily_pipeline", status=JobStatus.SUCCEEDED, stats={})
    videos = [
        Video(
            youtube_video_id=f"v{i}",
            channel_id=channel_id,
            title="t",
            description="",
            published_at=NOW,
            status=VideoStatus.ANALYZED,
        )
        for i in range(3)
    ]
    session.add_all([job, *videos])
    session.flush()
    return user, job.id, videos


def request(
    job: int, video: Video, model: str, result: LLMResult, ms: int, by: str | None = None
) -> LLMRequest:
    answered = by or (model if result is LLMResult.OK else None)
    return LLMRequest(
        job_run_id=job,
        video_id=video.id,
        model=model,
        answered_by=answered,
        provider="Acme" if answered else None,
        result=result,
        latency_ms=ms,
        prompt_tokens=1000 if result is LLMResult.OK else None,
        completion_tokens=200 if result is LLMResult.OK else None,
    )


def analysis(
    video: Video, summary: str, topics: list[str], user: User, raw_summary: str | None = None
) -> None:
    final: dict[str, Any] = {
        "one_line_summary": summary,
        "key_points": ["一", "二"],
        "key_concepts": [],
        "keywords": [],
    }
    raw = {**final, "one_line_summary": raw_summary or summary, "matched_topics": topics}
    raw |= {"relevance_score": 60, "limitation": None}
    session = Session.object_session(video)
    assert session is not None
    session.add(
        Analysis(
            video_id=video.id,
            model="m1",
            prompt_version="p",
            source_type=SourceType.AUTO_CAPTION,
            confidence_level=2,
            confidence_stars="⭐⭐",
            limitation=None,
            raw_response=raw,
            **final,
        )
    )
    session.add(
        UserVideoRelevance(user_id=user.id, video_id=video.id, relevance_score=60, matched_topics=topics)
    )
    session.flush()


@pytest.mark.db
def test_metrics_cover_latency_fallbacks_and_rule_checks(session: Session) -> None:
    user, job, (fast, fallback, failed) = setup(session)
    session.add_all(
        [
            request(job, fast, "m1", LLMResult.OK, 40_000),
            request(job, fallback, "m1", LLMResult.TIMEOUT, 180_000),
            request(job, fallback, "m2", LLMResult.OK, 60_000),
            request(job, failed, "m1", LLMResult.RATE_LIMITED, 500),
            request(job, failed, "m2", LLMResult.RATE_LIMITED, 400),
        ]
    )
    analysis(
        fast, "介紹 AI Agent。", ["AI Agent"], user, raw_summary="「介紹 AI Agent。」"
    )  # rules fixed quotes
    analysis(fallback, "使用大型語言模型 (LLM) 建立流程。", ["AI Agent", "RAG"], user)  # a violation left
    session.flush()

    metrics = collect_metrics(session, user, since=NOW - timedelta(days=1))
    m1, m2 = metrics.models
    assert (m2.model, m2.requests, m2.results) == ("m2", 2, {"ok": 1, "rate_limited": 1})
    assert (m1.model, m1.requests, m1.results) == ("m1", 3, {"ok": 1, "timeout": 1, "rate_limited": 1})
    assert (m1.p50_ms, m1.avg_prompt_tokens, m1.providers) == (40_000, 1000, {"Acme": 1})
    [day] = metrics.days
    assert (day.videos_tried, day.fallbacks, day.unanswered, day.timeouts) == (3, 1, 1, 1)
    assert (day.analyses, day.stars[2], day.topics_per_video) == (2, 2, 1.5)
    assert (day.rule_fixes, day.violations) == (1, 1)
    assert metrics.violations == [("v1", ["half-width gloss"])]

    text = "\n".join(render_metrics(metrics, tz=TZ))
    assert "requests 3: ok 1, timeout 1, rate_limited 1" in text
    assert "timeout rate 1/3 (33%), format error rate 0/3 (0%)" in text
    assert "fallback rate 1/2 (50%) of answered videos, unanswered 1/3 (33%), timeout rate 1/5 (20%)" in text
    assert "v1: half-width gloss" in text


@pytest.mark.db
def test_digest_warns_on_rerouted_models_and_rule_violations(session: Session) -> None:
    user, job, (video, *_) = setup(session)
    session.add(request(job, video, "vendor/m1:free", LLMResult.OK, 1000, by="vendor/other"))
    analysis(video, "使用大型語言模型 (LLM) 建立流程。", [], user)
    warnings = [w for w in trend_warnings(session, user, now=NOW) if "RSS" not in w]
    assert warnings[0].startswith("⚠️ 要求 vendor/m1:free 卻由 vendor/other 回應")
    assert warnings[1].startswith("⚠️ 1 部摘要仍有程式應修正的格式問題（half-width gloss 等）")


@pytest.mark.db
def test_no_invariant_warnings_when_answers_match_and_rules_hold(session: Session) -> None:
    user, job, (video, *_) = setup(session)
    session.add(request(job, video, "vendor/m1:free", LLMResult.OK, 1000, by="vendor/m1"))
    analysis(video, "介紹 AI Agent。", ["AI Agent"], user)
    warnings = [w for w in trend_warnings(session, user, now=NOW) if "RSS" not in w]
    assert warnings == []
