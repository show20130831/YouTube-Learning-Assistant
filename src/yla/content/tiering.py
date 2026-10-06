"""Choose what text a summary is based on and how much the reader can trust it.

Confidence comes from the source, never from the LLM:

    manual captions      -> 3 ⭐⭐⭐
    auto captions        -> 2 ⭐⭐
    title + description  -> 1 ⭐   (topic guess only, shown with a warning)
    not enough text      -> unavailable; no summary is generated
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from yla.content.cleaning import clean_description, clean_transcript
from yla.db.models import SourceType
from yla.youtube.transcripts import TranscriptResult, TranscriptStatus

STARS = {3: "⭐⭐⭐", 2: "⭐⭐", 1: "⭐"}

BASIS_LABEL = {
    SourceType.MANUAL_CAPTION: "人工字幕或完整逐字稿",
    SourceType.AUTO_CAPTION: "YouTube 自動字幕",
    SourceType.TITLE_DESCRIPTION: "影片標題與描述",
}

TOPIC_GUESS_WARNING = "⚠️ 目前沒有取得字幕，以下內容僅為主題推測，不代表完整影片摘要。"

_LEVEL = {SourceType.MANUAL_CAPTION: 3, SourceType.AUTO_CAPTION: 2, SourceType.TITLE_DESCRIPTION: 1}


class ContentTier(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_type: SourceType
    text: str  # what the LLM will read; empty when unavailable
    language: str | None = None

    @property
    def available(self) -> bool:
        return self.source_type is not SourceType.NONE

    @property
    def confidence_level(self) -> int:
        return _LEVEL.get(self.source_type, 0)

    @property
    def confidence_stars(self) -> str:
        return STARS.get(self.confidence_level, "")

    @property
    def basis_label(self) -> str:
        return BASIS_LABEL.get(self.source_type, "")

    @property
    def warning(self) -> str | None:
        return TOPIC_GUESS_WARNING if self.source_type is SourceType.TITLE_DESCRIPTION else None


def choose_content(
    transcript: TranscriptResult | None, title: str, description: str, *, min_description_chars: int
) -> ContentTier:
    if transcript is not None and transcript.has_text:
        cleaned = clean_transcript(transcript.text)
        if cleaned:
            source = (
                SourceType.MANUAL_CAPTION
                if transcript.status is TranscriptStatus.MANUAL
                else SourceType.AUTO_CAPTION
            )
            return ContentTier(source_type=source, text=cleaned, language=transcript.language)

    info = clean_description(description)
    combined = f"{title.strip()}\n{info}".strip()
    if info and len(combined) >= min_description_chars:
        return ContentTier(source_type=SourceType.TITLE_DESCRIPTION, text=combined)
    return ContentTier(source_type=SourceType.NONE, text="")
