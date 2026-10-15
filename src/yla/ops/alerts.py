"""Trend warnings for the daily digest: problems that persist, not one-off failures.

Rules use consecutive runs or days rather than rates: with 1-5 videos a day, a rate swings
wildly. Cloud feed failures are not alerted on by themselves: the cloud read is only a backup
for the home worker, and it fails routinely. What matters is whether *any* job managed to read
a channel, which ``subscriptions.last_checked_at`` already records.

Two more rules are invariants rather than trends, so a single occurrence is worth a warning: an
answer from a different model than the one requested (OpenRouter rerouting: the summaries may
change without any change on our side), and an output that still breaks a rule the post-processing
guarantees (a rule stopped working, or the model's output changed shape).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from yla.analysis.checks import GUARANTEED_BY_RULES
from yla.db.models import (
    Analysis,
    Channel,
    JobRun,
    JobStatus,
    LLMRequest,
    LLMResult,
    Subscription,
    User,
    Video,
)
from yla.ops.metrics import collect_metrics, same_model

STALE_CHANNEL_AFTER = timedelta(days=2)
PRIMARY_MODEL_RUNS = 3
CAPTION_ERROR_RUNS = 2
QUALITY_WINDOW = timedelta(days=7)
QUALITY_MIN_ANALYSES = 4
RULE_CHECK_WINDOW = timedelta(days=1)
DRIFT_CHECK_WINDOW = timedelta(days=8)  # weekly check plus a day of slack
# "Topic outside the list" is guaranteed when an analysis is made, but the list can change since
# (a topic removed from settings.yaml), so it is shown by ``yla metrics`` but not alerted on.
ALERT_ON = GUARANTEED_BY_RULES - {"topic outside the list"}


def trend_warnings(session: Session, user: User, *, now: datetime) -> list[str]:
    return [
        *_stale_channels(session, user, now),
        *_unreliable_primary_model(session),
        *_repeated_caption_errors(session),
        *_mostly_topic_guesses(session, user, now),
        *_rerouted_models(session),
        *_rule_violations(session, user, now),
        *_model_drift(session, now),
    ]


def _stale_channels(session: Session, user: User, now: datetime) -> list[str]:
    rows = session.execute(
        select(Channel.title, Channel.handle, Subscription.last_checked_at, Subscription.subscribed_at)
        .join(Subscription, Subscription.channel_id == Channel.id)
        .where(
            Subscription.user_id == user.id,
            Subscription.enabled.is_(True),
            Channel.youtube_channel_id.is_not(None),
        )
        .order_by(Channel.id)
    ).all()
    warnings = []
    for title, handle, last_checked, subscribed in rows:
        since = last_checked or subscribed  # never read: count from when it was added
        if now - since > STALE_CHANNEL_AFTER:
            days = (now - since).days
            warnings.append(f"⚠️ {title or handle} 已 {days} 天無法讀取 RSS，請確認頻道 ID 是否正確。")
    return warnings


def _recent_stats(session: Session, job_name: str, limit: int) -> list[dict[str, Any]]:
    return list(
        session.scalars(
            select(JobRun.stats)
            .where(JobRun.job_name == job_name, JobRun.status == JobStatus.SUCCEEDED)
            .order_by(JobRun.id.desc())
            .limit(limit)
        )
    )


def _unreliable_primary_model(session: Session) -> list[str]:
    runs = [s for s in _recent_stats(session, "daily_pipeline", 20) if s.get("analysis_attempts")]
    runs = runs[:PRIMARY_MODEL_RUNS]
    if len(runs) < PRIMARY_MODEL_RUNS:
        return []
    if all(2 * r.get("primary_failures", 0) >= r["analysis_attempts"] for r in runs):
        model = runs[0].get("primary_model") or "主模型"
        return [
            f"⚠️ {model} 最近 {PRIMARY_MODEL_RUNS} 次執行都有一半以上失敗（改用備援模型），"
            "建議在 settings.yaml 更換 primary_model。"
        ]
    return []


def _repeated_caption_errors(session: Session) -> list[str]:
    runs = _recent_stats(session, "local_worker", CAPTION_ERROR_RUNS)
    if len(runs) == CAPTION_ERROR_RUNS and all(r.get("errors", 0) > 0 and not r.get("blocked") for r in runs):
        return [
            f"⚠️ 本機 worker 連續 {CAPTION_ERROR_RUNS} 次抓字幕出錯，"
            "可能需要更新 youtube-transcript-api（uv lock --upgrade-package youtube-transcript-api）。"
        ]
    return []


def _mostly_topic_guesses(session: Session, user: User, now: datetime) -> list[str]:
    total, guesses = session.execute(
        select(func.count(Analysis.id), func.count(Analysis.id).filter(Analysis.confidence_level == 1))
        .join(Video, Video.id == Analysis.video_id)
        .join(Subscription, (Subscription.channel_id == Video.channel_id) & (Subscription.user_id == user.id))
        .where(Analysis.created_at >= now - QUALITY_WINDOW)
    ).one()
    if total >= QUALITY_MIN_ANALYSES and 2 * guesses > total:
        return [f"⚠️ 近 7 天 {guesses}/{total} 部摘要沒有字幕（⭐），請確認本機 worker 是否每天執行。"]
    return []


def _rerouted_models(session: Session) -> list[str]:
    last_run = session.scalar(
        select(func.max(JobRun.id)).where(
            JobRun.job_name == "daily_pipeline", JobRun.status == JobStatus.SUCCEEDED
        )
    )
    if last_run is None:
        return []
    answered = session.execute(
        select(LLMRequest.model, LLMRequest.answered_by)
        .where(LLMRequest.job_run_id == last_run, LLMRequest.result == LLMResult.OK)
        .distinct()
    ).all()
    return [
        f"⚠️ 要求 {model} 卻由 {by} 回應（OpenRouter 改道），摘要品質可能改變，可用 yla metrics 追蹤。"
        for model, by in answered
        if not same_model(model, by)
    ]


def _rule_violations(session: Session, user: User, now: datetime) -> list[str]:
    """Only problems the rules rule out: the prompt-only ones are trends for ``yla metrics``."""
    violations = [
        (video, broken)
        for video, problems in collect_metrics(session, user, since=now - RULE_CHECK_WINDOW).violations
        if (broken := [p for p in problems if p in ALERT_ON])
    ]
    if not violations:
        return []
    return [
        f"⚠️ {len(violations)} 部摘要仍有程式應修正的格式問題（{violations[0][1][0]} 等），"
        "規則可能失效或模型輸出格式改變，請執行 yla metrics 檢查。"
    ]


def _model_drift(session: Session, now: datetime) -> list[str]:
    """Warnings from the weekly drift check (``yla eval drift``).

    Derived from its job runs on every digest, never stored: a drift warning shows while the latest
    completed check found drift and disappears as soon as a later check is stable. A check stopped
    by the quota, skipped or failed does not count as done; if none completed for longer than the
    window, that is a warning too, so a check that silently stopped running cannot look like
    "no drift". Nothing is said before the first check ever ran (the weekly task not set up yet).
    """
    jobs = session.execute(
        select(JobRun.started_at, JobRun.status, JobRun.stats)
        .where(JobRun.job_name == "eval_drift")
        .order_by(JobRun.id.desc())
        .limit(30)
    ).all()
    if not jobs:
        return []
    completed = next(
        (
            (started, stats)
            for started, status, stats in jobs
            if status is JobStatus.SUCCEEDED and stats.get("outcome") in ("stable", "drifted")
        ),
        None,
    )
    warnings = []
    if completed is None or now - completed[0] > DRIFT_CHECK_WINDOW:
        _, status, stats = jobs[0]
        last = (
            "上次執行失敗"
            if status is JobStatus.FAILED
            else {
                "skipped": "上次因主模型或 prompt 已更換而跳過，需在 eval/drift.yaml 設定新的基準",
                "incomplete": "上次被免費額度擋下而未完成",
            }.get(stats.get("outcome"), "上次執行沒有完成")
        )
        when = "從未完成" if completed is None else f"已 {(now - completed[0]).days} 天沒有完成"
        warnings.append(f"⚠️ 每週漂移檢查{when}（{last}），請確認每週排程與電腦是否開機。")
    elif completed[1].get("outcome") == "drifted":
        stats = completed[1]
        reasons = "；".join([*stats.get("identity", []), *stats.get("quality", [])])
        warnings.append(
            f"⚠️ 每週漂移檢查：主模型和基準不同（{reasons}），模型可能已在同一名稱下更換，"
            f"請查看 eval/runs/{stats.get('run_id')}/drift.md。"
        )
    return warnings
