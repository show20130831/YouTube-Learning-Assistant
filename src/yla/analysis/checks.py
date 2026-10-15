"""Checks on a finished analysis that the post-processing rules are meant to guarantee.

Used by the evaluation (on replayed outputs) and by production monitoring (on stored analyses):
any residual violation means a rule stopped working or the model's output changed shape.
"""

from __future__ import annotations

import re

from yla.llm.schemas import MAX_KEY_POINTS, MAX_KEY_POINTS_TOPIC_GUESS, VideoAnalysis

# Problems the rules are meant to remove; any left in the final output is a residual violation.
_HALF_WIDTH_GLOSS = re.compile(r"[㐀-鿿]\s*\(\s*[A-Za-z]")
_REPEATED_GLOSS = re.compile(r"([A-Za-z][A-Za-z0-9 .\-]*)（\s*\1\s*）", re.IGNORECASE)
_WRAPPING_QUOTES = (("「", "」"), ("“", "”"), ('"', '"'), ("『", "』"))


# Problems the post-processing itself rules out. Any of these in a stored output means a rule broke
# or the output changed shape; the others are only asked of the model by the prompt.
GUARANTEED_BY_RULES = frozenset(
    {
        "half-width gloss",
        "repeated gloss",
        "wrapping quotes",
        "too many key points",
        "topic outside the list",
        "score out of range",
    }
)

# The topic-guess tier (title and description only) must read as a guess; the prompt asks for these.
SPECULATIVE_MARKERS = ("可能", "推測", "或許", "似乎", "應該", "看起來", "疑似", "大概")


def residual_violations(
    analysis: VideoAnalysis, topic_names: set[str], *, topic_guess: bool = False
) -> list[str]:
    texts = [analysis.one_line_summary, *analysis.key_points]
    problems = []
    if any(_HALF_WIDTH_GLOSS.search(t) for t in texts):
        problems.append("half-width gloss")
    if any(_REPEATED_GLOSS.search(t) for t in texts):
        problems.append("repeated gloss")
    if any(t.startswith(left) and t.endswith(right) for t in texts for left, right in _WRAPPING_QUOTES):
        problems.append("wrapping quotes")
    if not analysis.key_points:
        problems.append("no key points")
    if len(analysis.key_points) > (MAX_KEY_POINTS_TOPIC_GUESS if topic_guess else MAX_KEY_POINTS):
        problems.append("too many key points")
    if topic_guess and not any(m in analysis.one_line_summary for m in SPECULATIVE_MARKERS):
        problems.append("summary not worded as a guess")
    if set(analysis.matched_topics) - topic_names:
        problems.append("topic outside the list")
    if not 0 <= analysis.relevance_score <= 100:
        problems.append("score out of range")
    return problems


def fix_count(raw: VideoAnalysis, final: VideoAnalysis) -> int:
    """How many places the rules changed the model's text (summary, points, trimmed points)."""
    changed = int(raw.one_line_summary.strip() != final.one_line_summary)
    raw_points = [p.strip() for p in raw.key_points]
    changed += sum(1 for p in final.key_points if p not in raw_points)
    changed += max(0, len(raw.key_points) - len(final.key_points))
    return changed
