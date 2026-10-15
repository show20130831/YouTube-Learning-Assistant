"""Analyse one video with the LLM and store the result.

Model order comes from settings (primary, then fallbacks). Per model:
- a rate limit or provider error moves on to the next model;
- an invalid reply is retried once on the same model with the validation error attached.

Every request counts against the free daily quota, so ``calls`` is reported for job stats.
Map-reduce for very long transcripts is a Phase 2 item; until then over-long text is cut and
the prompt asks the model to state that only the first part was analysed.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import ValidationError
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from yla.analysis.topics import TopicMatcher
from yla.config import TopicConfig
from yla.content.tiering import ContentTier
from yla.db.models import Analysis, SourceType, UserVideoRelevance, Video, VideoStatus
from yla.llm.client import (
    ChatMessage,
    LLMClient,
    LLMError,
    LLMOutputInvalid,
    LLMRateLimited,
    LLMResponse,
    LLMTimeout,
)
from yla.llm.prompts import PROMPT_VERSION, build_messages
from yla.llm.schemas import ANALYSIS_JSON_SCHEMA, ANALYSIS_SCHEMA_NAME, VideoAnalysis, normalize

logger = logging.getLogger(__name__)


# Per-request results, recorded for operations metrics (which models fail, and how).
Attempt = tuple[str, str]  # (model, "ok" | "timeout" | "rate_limited" | "invalid" | "error")


@dataclass(frozen=True)
class RequestLog:
    """One LLM request, for monitoring: who answered, how, how fast and at what size."""

    model: str  # the model requested
    result: str  # "ok" | "timeout" | "rate_limited" | "invalid" | "error"
    latency_ms: int
    answered_by: str | None = None  # the model that actually replied (OpenRouter may reroute)
    provider: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _request_log(
    model: str, result: str, started: float, clock: Callable[[], float], response: LLMResponse | None
) -> RequestLog:
    usage = response.usage if response else {}
    return RequestLog(
        model=model,
        result=result,
        latency_ms=round((clock() - started) * 1000),
        answered_by=response.model if response else None,
        provider=response.provider if response else None,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def _attempt_result(exc: Exception) -> str:
    if isinstance(exc, LLMTimeout):
        return "timeout"
    if isinstance(exc, LLMRateLimited):
        return "rate_limited"
    if isinstance(exc, LLMOutputInvalid | ValidationError):
        return "invalid"
    return "error"


class AnalysisFailed(Exception):
    def __init__(
        self,
        errors: dict[str, str],
        calls: int,
        *,
        rate_limited: bool = False,
        attempts: list[Attempt] | None = None,
        requests: list[RequestLog] | None = None,
    ) -> None:
        super().__init__("; ".join(f"{model}: {error}" for model, error in errors.items()))
        self.errors = errors
        self.calls = calls
        # Every model answered 429: the free daily quota is most likely used up for today.
        self.rate_limited = rate_limited
        self.attempts = attempts or []
        self.requests = requests or []


@dataclass(frozen=True)
class AnalysisOutcome:
    analysis: VideoAnalysis
    tier: ContentTier
    model: str
    prompt_version: str
    raw: dict[str, object]
    calls: int
    errors: dict[str, str] = field(default_factory=dict)  # models that failed before one succeeded
    attempts: list[Attempt] = field(default_factory=list)
    primary_ok: bool = True  # the first configured model produced the result
    requests: list[RequestLog] = field(default_factory=list)


MIN_SCORE_WITH_TOPICS = 40
MAX_SCORE_WITHOUT_TOPICS = 20

# Version of the deterministic post-processing (``normalize``, ``tidy_text``, ``TopicMatcher`` and
# ``reconcile_topics``). Bump it whenever their behaviour changes, so evaluation runs are compared
# only when their outputs were cleaned up the same way.
RULES_VERSION = "2026-10-08.1"


def reconcile_topics(analysis: VideoAnalysis, title: str, matcher: TopicMatcher) -> VideoAnalysis:
    """Add topics found deterministically in keywords, concepts and the title, and keep the score
    consistent with them. Free models pick topics inconsistently between runs."""
    found = [
        *matcher.from_terms([*analysis.keywords, *(c.term for c in analysis.key_concepts)]),
        *matcher.from_text(title),
    ]
    topics = list(dict.fromkeys([*analysis.matched_topics, *found]))
    score = analysis.relevance_score
    score = max(score, MIN_SCORE_WITH_TOPICS) if topics else min(score, MAX_SCORE_WITHOUT_TOPICS)
    return analysis.model_copy(update={"matched_topics": topics, "relevance_score": score})


class Analyzer:
    def __init__(
        self,
        client: LLMClient,
        models: Sequence[str],
        topics: Sequence[TopicConfig],
        *,
        max_input_chars: int = 400_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not models:
            raise ValueError("no LLM model configured: set llm.primary_model in settings.yaml")
        self._client = client
        self._models = list(models)
        self._topics = list(topics)
        self._matcher = TopicMatcher(topics)
        self._max_input_chars = max_input_chars
        self._clock = clock

    def analyze(self, *, title: str, channel: str, tier: ContentTier) -> AnalysisOutcome:
        if not tier.available:
            raise ValueError("cannot analyse a video without usable text")

        truncated = len(tier.text) > self._max_input_chars
        if truncated:
            tier = tier.model_copy(update={"text": tier.text[: self._max_input_chars]})
        messages = build_messages(
            title=title, channel=channel, tier=tier, topics=self._topics, truncated=truncated
        )

        calls = 0
        errors: dict[str, str] = {}
        rate_limited: set[str] = set()
        attempts: list[Attempt] = []
        requests: list[RequestLog] = []
        for model in self._models:
            attempt_messages = list(messages)
            for attempt in (1, 2):
                calls += 1
                started = self._clock()
                response: LLMResponse | None = None
                try:
                    response = self._client.complete_json(
                        attempt_messages,
                        schema=ANALYSIS_JSON_SCHEMA,
                        schema_name=ANALYSIS_SCHEMA_NAME,
                        model=model,
                    )
                    parsed = VideoAnalysis.model_validate(response.data)
                except (LLMOutputInvalid, ValidationError) as exc:
                    attempts.append((model, _attempt_result(exc)))
                    requests.append(_request_log(model, "invalid", started, self._clock, response))
                    errors[model] = f"invalid output: {str(exc)[:200]}"
                    if attempt == 1:
                        attempt_messages = [*messages, _fix_request(exc)]
                        continue
                    break
                except LLMError as exc:
                    attempts.append((model, _attempt_result(exc)))
                    requests.append(_request_log(model, _attempt_result(exc), started, self._clock, None))
                    errors[model] = f"{type(exc).__name__}: {exc}"
                    if isinstance(exc, LLMRateLimited):
                        rate_limited.add(model)
                    break
                requests.append(_request_log(model, "ok", started, self._clock, response))

                analysis = normalize(
                    parsed,
                    topic_lookup=self._matcher.lookup,
                    topic_guess=tier.source_type is SourceType.TITLE_DESCRIPTION,
                )
                analysis = reconcile_topics(analysis, title, self._matcher)
                if truncated and not analysis.limitation:
                    analysis = analysis.model_copy(update={"limitation": "內容過長，只分析了前段。"})
                errors.pop(model, None)
                attempts.append((model, "ok"))
                return AnalysisOutcome(
                    analysis=analysis,
                    tier=tier,
                    model=response.model,
                    prompt_version=PROMPT_VERSION,
                    raw=response.data,
                    calls=calls,
                    errors=errors,
                    attempts=attempts,
                    primary_ok=model == self._models[0],
                    requests=requests,
                )
            logger.warning("model %s failed: %s", model, errors.get(model))
        raise AnalysisFailed(
            errors,
            calls,
            rate_limited=rate_limited == set(self._models),
            attempts=attempts,
            requests=requests,
        )


def _fix_request(error: Exception) -> ChatMessage:
    return {
        "role": "user",
        "content": f"上一個回覆不符合 JSON schema（{str(error)[:300]}）。請重新輸出完整、有效的 JSON。",
    }


def store_analysis(session: Session, *, video: Video, user_id: int, outcome: AnalysisOutcome) -> Analysis:
    """Save the analysis (a new row each time, so re-analysis keeps history) and mark the video done."""
    result = outcome.analysis
    row = Analysis(
        video_id=video.id,
        model=outcome.model,
        prompt_version=outcome.prompt_version,
        source_type=outcome.tier.source_type,
        confidence_level=outcome.tier.confidence_level,
        confidence_stars=outcome.tier.confidence_stars,
        limitation=result.limitation,
        one_line_summary=result.one_line_summary,
        key_points=result.key_points,
        key_concepts=[c.model_dump() for c in result.key_concepts],
        keywords=result.keywords,
        raw_response=outcome.raw,
    )
    session.add(row)

    values = {"relevance_score": result.relevance_score, "matched_topics": result.matched_topics}
    session.execute(
        insert(UserVideoRelevance)
        .values(user_id=user_id, video_id=video.id, **values)
        .on_conflict_do_update(index_elements=["user_id", "video_id"], set_=values)
    )
    video.status = VideoStatus.ANALYZED
    video.last_error = None
    session.flush()
    return row
