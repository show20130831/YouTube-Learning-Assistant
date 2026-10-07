"""``yla demo``: the content pipeline end to end with no accounts, network or database.

A fictional channel's feed goes through the real code (feed parsing, Shorts filter, caption
cleaning and confidence tiers, analysis with code-side fixes, topic matching, LINE formatting).
Only the outside world is replaced: video lengths stand in for the YouTube Data API, stored
captions for the local worker, and recorded replies for the LLM. Storage and scheduling are
skipped; see the README for how the deployed system runs them.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from importlib.resources import files
from typing import Any

from yla.analysis.analyzer import Analyzer
from yla.analysis.topics import TopicMatcher
from yla.config import TopicConfig
from yla.content.cleaning import clean_transcript
from yla.content.tiering import choose_content
from yla.llm.client import ChatMessage, LLMResponse
from yla.notify.formatter import DigestData, DigestVideo, build_messages
from yla.pipeline.enrich import SHORTS_MAX_SECONDS
from yla.youtube.rss import parse_feed
from yla.youtube.transcripts import TranscriptResult, TranscriptStatus

DEMO_MODEL = "recorded-demo-replies"
DEMO_TOPICS = [
    TopicConfig(name="AI Agent", aliases=["Agent", "Agents", "Agentic AI", "AI Agents"]),
    TopicConfig(name="LLM", aliases=["Large Language Model", "Large Language Models"]),
    TopicConfig(name="RAG", aliases=["Retrieval-Augmented Generation", "Retrieval Augmented Generation"]),
    TopicConfig(name="LangChain"),
    TopicConfig(name="LangGraph"),
    TopicConfig(name="FastAPI"),
    TopicConfig(name="Machine Learning", aliases=["ML"]),
]
_DATA = files("yla.demo") / "data"


@dataclass
class DemoResult:
    messages: list[str]
    explanation: list[str] = field(default_factory=list)


class RecordedLLM:
    """Stands in for OpenRouter: returns the recorded reply for the video in the prompt."""

    def __init__(self, replies: dict[str, dict[str, Any]]) -> None:
        self._replies = replies

    def complete_json(
        self, messages: list[ChatMessage], *, schema: dict[str, Any], schema_name: str, model: str
    ) -> LLMResponse:
        title = messages[1]["content"].splitlines()[0].removeprefix("影片標題：")
        return LLMResponse(data=self._replies[title], model=model)


def _load_json(name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((_DATA / name).read_text(encoding="utf-8"))
    return data


def run_demo(today: date | None = None) -> DemoResult:
    feed = parse_feed((_DATA / "feed.xml").read_bytes())
    videos_data = _load_json("videos.json")
    replies = _load_json("llm_replies.json")
    analyzer = Analyzer(RecordedLLM(replies), [DEMO_MODEL], DEMO_TOPICS)
    matcher = TopicMatcher(DEMO_TOPICS)
    channel = feed.channel_title or "Demo channel"

    explanation: list[str] = []
    videos: list[DigestVideo] = []
    unavailable: list[str] = []
    shorts = 0

    for entry in feed.entries:
        say = explanation.append
        say(f"[{entry.video_id}] {entry.title}")
        duration = videos_data["metadata"][entry.video_id]["duration_seconds"]
        if duration <= SHORTS_MAX_SECONDS:
            shorts += 1
            say(f"  skipped: {duration}s long, a Short (<= {SHORTS_MAX_SECONDS}s)")
            continue

        raw = videos_data["transcripts"].get(entry.video_id)
        transcript = (
            TranscriptResult(
                status=TranscriptStatus(raw["status"]), language=raw["language"], text=raw["text"]
            )
            if raw
            else None
        )
        tier = choose_content(transcript, entry.title, entry.description, min_description_chars=150)
        if transcript is not None:
            before, after = len(transcript.text), len(clean_transcript(transcript.text))
            say(f"  captions: {transcript.status.value}, {before} -> {after} chars after cleaning")
        else:
            say("  captions: none, falling back to title + description")
        if not tier.available:
            unavailable.append(f"{entry.title}｜{channel}")
            say("  unavailable: too little text to summarise, so no summary is generated")
            continue
        say(
            f"  confidence: {tier.confidence_stars} ({tier.source_type.value}), set by code, not by the model"
        )

        outcome = analyzer.analyze(title=entry.title, channel=channel, tier=tier)
        result, reply = outcome.analysis, replies[entry.title]
        model_topics = matcher.from_terms(reply["matched_topics"])
        added = [t for t in result.matched_topics if t not in model_topics]
        say(
            f"  topics: model picked {model_topics or 'none'}; added by TopicMatcher: {added or 'none'}; "
            f"relevance {reply['relevance_score']} -> {result.relevance_score}"
        )
        fixes = [
            f"summary: {reply['one_line_summary']!r} -> {result.one_line_summary!r}"
            for _ in [0]
            if reply["one_line_summary"] != result.one_line_summary
        ] + [
            f"point: {before!r} -> {after!r}"
            for before, after in zip(reply["key_points"], result.key_points, strict=False)
            if before != after
        ]
        say(f"  text fixes: {len(fixes) or 'none'}")
        explanation.extend(f"    {fix}" for fix in fixes)

        videos.append(
            DigestVideo(
                title=entry.title,
                channel=channel,
                url=entry.url,
                source_type=tier.source_type,
                confidence_level=tier.confidence_level,
                one_line_summary=result.one_line_summary,
                key_points=result.key_points,
                keywords=result.keywords,
                matched_topics=result.matched_topics,
                relevance_score=result.relevance_score,
            )
        )

    trends = Counter(topic for v in videos for topic in v.matched_topics)
    digest = DigestData(
        day=today or date.today(),
        channels_tracked=1,
        new_videos=len(feed.entries) - shorts,
        videos=videos,
        shorts_skipped=shorts,
        unavailable=unavailable,
        trends=sorted(trends.items(), key=lambda kv: (-kv[1], kv[0]))[:5],
    )
    return DemoResult(messages=build_messages(digest), explanation=explanation)
