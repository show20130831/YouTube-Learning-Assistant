"""Text cleanup before analysis and before measuring whether a description is informative."""

from __future__ import annotations

import re

# Non-speech caption tags: [Music], [Applause], (laughter), ♪ ...
_CAPTION_TAG = re.compile(
    r"\[[^\]]{1,40}\]|\((?:music|applause|laughter|laughs|inaudible)\)|♪+", re.IGNORECASE
)
_SPEAKER_MARK = re.compile(r"^\s*>>\s*")
_WS = re.compile(r"[^\S\n]+")  # any whitespace except newlines, incl. no-break space

_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HASHTAG = re.compile(r"(?<!\w)#[\w-]+")
_PROMO_LINE = re.compile(
    r"\b(subscribe|follow (?:us|me)|twitter|instagram|linkedin|tiktok|discord|patreon|facebook|"
    r"newsletter|sponsor(?:ed)?|merch|affiliate|join (?:this|our) channel)\b",
    re.IGNORECASE,
)
# A line that is only a label once its URL is gone, e.g. "→ Docs:" or "- GitHub:".
_LABEL_ONLY = re.compile(r"^[\W_]*[\w .'/&-]{0,30}:?\s*[\W_]*$")


def clean_transcript(text: str) -> str:
    """Drop caption tags and speaker marks, and the consecutive duplicate lines auto captions repeat."""
    lines: list[str] = []
    for raw in text.splitlines():
        line = _WS.sub(" ", _CAPTION_TAG.sub(" ", _SPEAKER_MARK.sub("", raw))).strip()
        if line and (not lines or line != lines[-1]):
            lines.append(line)
    return " ".join(lines)


def clean_description(text: str) -> str:
    """Keep the informative part of a description: drop links, hashtags and promo lines.

    Chapter lines ("2:10 State management") are kept: they outline the video's content.
    """
    kept: list[str] = []
    for raw in text.splitlines():
        had_url = bool(_URL.search(raw))
        line = _WS.sub(" ", _HASHTAG.sub("", _URL.sub("", raw))).strip()
        if not line:
            continue
        if _PROMO_LINE.search(line) or (had_url and _LABEL_ONLY.match(line)):
            continue
        kept.append(line)
    return "\n".join(kept)
