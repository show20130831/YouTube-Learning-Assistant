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
from collections.abc import Sequence
from dataclasses import dataclass, field

from pydantic import ValidationError
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from yla.analysis.topics import TopicMatcher
from yla.config import TopicConfig
from yla.content.tiering import ContentTier
from yla.db.models import Analysis, SourceType, UserVideoRelevance, Video, VideoStatus
from yla.llm.client import ChatMessage, LLMClient, LLMError, LLMOutputInvalid
from yla.llm.prompts import PROMPT_VERSION, build_messages
from yla.llm.schemas import ANALYSIS_JSON_SCHEMA, ANALYSIS_SCHEMA_NAME, VideoAnalysis, normalize

logger = logging.getLogger(__name__)


class AnalysisFailed(Exception):
    def __init__(self, errors: dict[str, str], calls: int) -> None:
        super().__init__("; ".join(f"{model}: {error}" for model, error in errors.items()))
        self.errors = errors
        self.calls = calls


@dataclass(frozen=True)
class AnalysisOutcome:
    analysis: VideoAnalysis
    tier: ContentTier
    model: str
    prompt_version: str
    raw: dict[str, object]
    calls: int
    errors: dict[str, str] = field(default_factory=dict)  # models that failed before one succeeded


MIN_SCORE_WITH_TOPICS = 40
MAX_SCORE_WITHOUT_TOPICS = 20


class Analyzer:
    def __init__(
        self,
        client: LLMClient,
        models: Sequence[str],
        topics: Sequence[TopicConfig],
        *,
        max_input_chars: int = 400_000,
    ) -> None:
        if not models:
            raise ValueError("no LLM model configured: set llm.primary_model in settings.yaml")
        self._client = client
        self._models = list(models)
        self._topics = list(topics)
        self._matcher = TopicMatcher(topics)
        self._max_input_chars = max_input_chars

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
        for model in self._models:
            attempt_messages = list(messages)
            for attempt in (1, 2):
                calls += 1
                try:
                    response = self._client.complete_json(
                        attempt_messages,
                        schema=ANALYSIS_JSON_SCHEMA,
                        schema_name=ANALYSIS_SCHEMA_NAME,
                        model=model,
                    )
                    parsed = VideoAnalysis.model_validate(response.data)
                except (LLMOutputInvalid, ValidationError) as exc:
                    errors[model] = f"invalid output: {str(exc)[:200]}"
                    if attempt == 1:
                        attempt_messages = [*messages, _fix_request(exc)]
                        continue
                    break
                except LLMError as exc:
                    errors[model] = f"{type(exc).__name__}: {exc}"
                    break

                analysis = normalize(
                    parsed,
                    topic_lookup=self._matcher.lookup,
                    topic_guess=tier.source_type is SourceType.TITLE_DESCRIPTION,
                )
                analysis = self._reconcile_topics(analysis, title)
                if truncated and not analysis.limitation:
                    analysis = analysis.model_copy(update={"limitation": "內容過長，只分析了前段。"})
                errors.pop(model, None)
                return AnalysisOutcome(
                    analysis=analysis,
                    tier=tier,
                    model=response.model,
                    prompt_version=PROMPT_VERSION,
                    raw=response.data,
                    calls=calls,
                    errors=errors,
                )
            logger.warning("model %s failed: %s", model, errors.get(model))
        raise AnalysisFailed(errors, calls)

    def _reconcile_topics(self, analysis: VideoAnalysis, title: str) -> VideoAnalysis:
        """Add topics found deterministically in keywords, concepts and the title, and keep the
        score consistent with them. Free models pick topics inconsistently between runs."""
        found = [
            *self._matcher.from_terms([*analysis.keywords, *(c.term for c in analysis.key_concepts)]),
            *self._matcher.from_text(title),
        ]
        topics = list(dict.fromkeys([*analysis.matched_topics, *found]))
        score = analysis.relevance_score
        score = max(score, MIN_SCORE_WITH_TOPICS) if topics else min(score, MAX_SCORE_WITHOUT_TOPICS)
        return analysis.model_copy(update={"matched_topics": topics, "relevance_score": score})


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
