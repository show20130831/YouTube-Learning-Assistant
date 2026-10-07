"""Deterministic topic matching against the user's topic names and aliases.

Used to back up the LLM (free models pick ``matched_topics`` inconsistently: the same video got
"AI Agent" in one run and nothing in the next) and, later, to rank videos without spending LLM
requests when there are more new videos than the daily limit.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from yla.config import TopicConfig


class TopicMatcher:
    def __init__(self, topics: Sequence[TopicConfig]) -> None:
        self._lookup = {term.casefold(): t.name for t in topics for term in t.all_names()}
        # Longest terms first so "AI Agents" wins over "Agent" inside the same text.
        terms = sorted(self._lookup, key=len, reverse=True)
        self._pattern = (
            re.compile("|".join(_with_boundaries(t) for t in terms), re.IGNORECASE) if terms else None
        )

    @property
    def lookup(self) -> dict[str, str]:
        """Every topic name and alias (casefolded) -> canonical topic name."""
        return self._lookup

    def from_terms(self, terms: Iterable[str]) -> list[str]:
        """Topics whose name or alias equals one of the terms (e.g. LLM keywords)."""
        return _unique(
            self._lookup[t.strip().casefold()] for t in terms if t.strip().casefold() in self._lookup
        )

    def from_text(self, text: str) -> list[str]:
        """Topics whose name or alias appears as a whole word or phrase in free text."""
        if self._pattern is None:
            return []
        return _unique(self._lookup[m.group(0).casefold()] for m in self._pattern.finditer(text))


_ASCII_WORD = re.compile(r"[A-Za-z0-9]")


def _with_boundaries(term: str) -> str:
    """Whole-word only where the term has an English letter or digit at that edge.

    "ML" must not match inside "HTML", but Chinese has no spaces between words, so
    "機器學習" must match in "機器學習2026" and "LangChain" in "用LangChain打造". Python's regex
    word classes treat Chinese characters as letters, so they cannot be used for this.
    """
    pattern = re.escape(term)
    if _ASCII_WORD.match(term[0]):
        pattern = r"(?<![A-Za-z0-9_-])" + pattern
    if _ASCII_WORD.match(term[-1]):
        pattern += r"(?![A-Za-z0-9_-])"
    return pattern


def _unique(items: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(items))
