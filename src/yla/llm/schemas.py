"""The structured output asked from the LLM, and its cleanup.

The JSON schema sent to the model is written by hand (strict mode: every field required, no
extra properties) and kept lenient on counts; ``normalize`` enforces limits afterwards, so a
model that returns 7 key points costs a trim, not another request against the free quota.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

MAX_KEY_POINTS = 6
MAX_KEY_POINTS_TOPIC_GUESS = 3
MAX_CONCEPTS = 8
MAX_KEYWORDS = 8


class Concept(BaseModel):
    model_config = ConfigDict(frozen=True)

    term: str
    explanation: str


class VideoAnalysis(BaseModel):
    """What the LLM returns. Confidence and source are added by code, never by the model."""

    model_config = ConfigDict(frozen=True)

    one_line_summary: str = Field(min_length=1)
    key_points: list[str]
    key_concepts: list[Concept]
    keywords: list[str]
    matched_topics: list[str]
    relevance_score: int
    limitation: str | None


_STRING_LIST: dict[str, Any] = {"type": "array", "items": {"type": "string"}}

ANALYSIS_SCHEMA_NAME = "video_analysis"
ANALYSIS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "one_line_summary",
        "key_points",
        "key_concepts",
        "keywords",
        "matched_topics",
        "relevance_score",
        "limitation",
    ],
    "properties": {
        "one_line_summary": {"type": "string"},
        "key_points": _STRING_LIST,
        "key_concepts": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["term", "explanation"],
                "properties": {"term": {"type": "string"}, "explanation": {"type": "string"}},
            },
        },
        "keywords": _STRING_LIST,
        "matched_topics": _STRING_LIST,
        "relevance_score": {"type": "integer"},
        "limitation": {"type": ["string", "null"]},
    },
}


def _clean_list(items: list[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        text = item.strip()
        if text and text.casefold() not in seen:
            seen.add(text.casefold())
            out.append(text)
    return out[:limit]


def normalize(analysis: VideoAnalysis, *, topic_lookup: dict[str, str], topic_guess: bool) -> VideoAnalysis:
    """Trim and dedupe, clamp the score, and keep only topics from the user's list.

    ``topic_lookup`` maps every topic name and alias (casefolded) to the canonical topic name.
    """
    concepts: list[Concept] = []
    seen_terms: set[str] = set()
    for concept in analysis.key_concepts:
        term = concept.term.strip()
        if term and term.casefold() not in seen_terms and len(concepts) < MAX_CONCEPTS:
            seen_terms.add(term.casefold())
            concepts.append(Concept(term=term, explanation=concept.explanation.strip()))

    topics = [
        topic_lookup[t.strip().casefold()]
        for t in analysis.matched_topics
        if t.strip().casefold() in topic_lookup
    ]
    limitation = (analysis.limitation or "").strip() or None
    return VideoAnalysis(
        one_line_summary=analysis.one_line_summary.strip(),
        key_points=_clean_list(
            analysis.key_points, MAX_KEY_POINTS_TOPIC_GUESS if topic_guess else MAX_KEY_POINTS
        ),
        key_concepts=concepts,
        keywords=_clean_list(analysis.keywords, MAX_KEYWORDS),
        matched_topics=_clean_list(topics, len(topics)),
        relevance_score=max(0, min(100, analysis.relevance_score)),
        limitation=limitation,
    )
