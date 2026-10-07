"""The structured output asked from the LLM, and its cleanup.

The JSON schema sent to the model is written by hand (strict mode: every field required, no
extra properties) and kept lenient on counts; ``normalize`` enforces limits afterwards, so a
model that returns 7 key points costs a trim, not another request against the free quota.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# A mechanical "可能 " / "推測：" prefix on topic-guess points: the warning shown above them already
# says the content is a guess. Natural phrasing such as "可能介紹……" has no separator and is kept.
_GUESS_PREFIX = re.compile(r"^(?:可能|推測)[\s:：,，、]+")

_CJK = "㐀-鿿"
# An English term followed by a full-width bracket gloss, e.g. "AI Agent（AI Agent）". The
# lookahead captures the next character so a space can be kept before following Chinese text.
_ENGLISH_WITH_GLOSS = re.compile(rf"([A-Za-z][A-Za-z0-9 .\-]*)（([^（）]+)）(?=([{_CJK}])?)")


def drop_repeated_gloss(text: str) -> str:
    """ "AI Agent（AI Agent）在" -> "AI Agent 在". Real glosses like "技能（Skill）" are left alone."""

    def replace(match: re.Match[str]) -> str:
        term, gloss = match.group(1), match.group(2).strip()
        if not term.strip().casefold().endswith(gloss.casefold()):
            return match.group(0)
        return f"{term} " if match.group(3) else term

    return _ENGLISH_WITH_GLOSS.sub(replace, text)


# "大型語言模型 (Large Language Model) 呼叫" -> "大型語言模型（Large Language Model）呼叫"
_HALF_WIDTH_GLOSS = re.compile(rf"([{_CJK}])\s*\(\s*([A-Za-z][^()]*?)\s*\)(?:\s+(?=[{_CJK}]))?")
_WRAPPING_QUOTES = (("「", "」"), ("“", "”"), ('"', '"'), ("『", "』"))


def tidy_text(text: str) -> str:
    """Small, deterministic style fixes the prompt cannot guarantee with free models."""
    text = text.strip()
    for left, right in _WRAPPING_QUOTES:
        if len(text) > 2 and text.startswith(left) and text.endswith(right) and text.count(left) == 1:
            text = text[1:-1].strip()
    text = _HALF_WIDTH_GLOSS.sub(r"\1（\2）", text)
    return drop_repeated_gloss(text)


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
    points = [tidy_text(p) for p in analysis.key_points]
    if topic_guess:
        points = [_GUESS_PREFIX.sub("", p.strip()) for p in points]
    return VideoAnalysis(
        one_line_summary=tidy_text(analysis.one_line_summary),
        key_points=_clean_list(points, MAX_KEY_POINTS_TOPIC_GUESS if topic_guess else MAX_KEY_POINTS),
        key_concepts=concepts,
        keywords=_clean_list(analysis.keywords, MAX_KEYWORDS),
        matched_topics=_clean_list(topics, len(topics)),
        relevance_score=max(0, min(100, analysis.relevance_score)),
        limitation=limitation,
    )
