import pytest

from yla.content.cleaning import clean_description, clean_transcript
from yla.content.tiering import TOPIC_GUESS_WARNING, ContentTier, choose_content
from yla.db.models import SourceType
from yla.youtube.transcripts import TranscriptResult, TranscriptStatus

# --- transcript cleaning ---------------------------------------------------------------


def test_transcript_tags_and_speaker_marks_removed() -> None:
    raw = "[Music]\n>> welcome back\nto the channel [Applause]\n♪♪\n(laughter) okay"
    assert clean_transcript(raw) == "welcome back to the channel okay"


def test_consecutive_duplicate_lines_dropped_but_later_repeats_kept() -> None:
    raw = "so the agent\nso the agent\ncalls a tool\nso the agent"
    assert clean_transcript(raw) == "so the agent calls a tool so the agent"


def test_whitespace_collapsed() -> None:
    nbsp = chr(0xA0)
    assert clean_transcript(f"  state{nbsp}{nbsp}management \n\n  matters ") == "state management matters"


# --- description cleaning --------------------------------------------------------------

DESCRIPTION = """How to keep agent state consistent and handle tool-calling errors.

Chapters:
0:00 Why agents fail
2:10 State management

→ Docs: https://example.com/agents
Follow us on Twitter: https://x.com/example
Subscribe for more!
#ai #agents
Code at https://github.com/example/repo shows the full retry loop."""


def test_description_keeps_content_and_chapters() -> None:
    cleaned = clean_description(DESCRIPTION)
    assert cleaned.splitlines() == [
        "How to keep agent state consistent and handle tool-calling errors.",
        "Chapters:",
        "0:00 Why agents fail",
        "2:10 State management",
        "Code at shows the full retry loop.",
    ]


@pytest.mark.parametrize(
    "description", ["", "#rag #llm", "https://example.com", "Subscribe! https://y.tube/x"]
)
def test_description_with_no_information_cleans_to_empty(description: str) -> None:
    assert clean_description(description) == ""


# --- tiering ---------------------------------------------------------------------------

LONG_DESCRIPTION = "This video explains how retrieval-augmented generation grounds answers in documents. " * 3


def tier(
    transcript: TranscriptResult | None, description: str = LONG_DESCRIPTION, title: str = "RAG Basics"
) -> ContentTier:
    return choose_content(transcript, title, description, min_description_chars=150)


def test_manual_captions_are_three_stars() -> None:
    result = tier(
        TranscriptResult(status=TranscriptStatus.MANUAL, language="en", text="[Music]\nhello\nhello")
    )
    assert result.source_type is SourceType.MANUAL_CAPTION
    assert (result.confidence_level, result.confidence_stars) == (3, "⭐⭐⭐")
    assert result.basis_label == "人工字幕或完整逐字稿"
    assert result.text == "hello"
    assert result.language == "en"
    assert result.warning is None


def test_auto_captions_are_two_stars() -> None:
    result = tier(TranscriptResult(status=TranscriptStatus.GENERATED, language="en", text="hi there"))
    assert (result.source_type, result.confidence_stars) == (SourceType.AUTO_CAPTION, "⭐⭐")
    assert result.basis_label == "YouTube 自動字幕"


@pytest.mark.parametrize(
    "transcript",
    [
        None,
        TranscriptResult(status=TranscriptStatus.NONE),
        TranscriptResult(status=TranscriptStatus.BLOCKED, error="RequestBlocked"),
        TranscriptResult(status=TranscriptStatus.GENERATED, text="[Music]\n♪"),  # nothing left after cleaning
    ],
)
def test_without_usable_captions_falls_back_to_title_and_description(
    transcript: TranscriptResult | None,
) -> None:
    result = tier(transcript)
    assert result.source_type is SourceType.TITLE_DESCRIPTION
    assert (result.confidence_level, result.confidence_stars) == (1, "⭐")
    assert result.basis_label == "影片標題與描述"
    assert result.warning == TOPIC_GUESS_WARNING
    assert result.text.startswith("RAG Basics\n")


@pytest.mark.parametrize("description", ["", "#rag #llm https://example.com", "Short blurb."])
def test_too_little_text_is_unavailable(description: str) -> None:
    result = tier(None, description=description)
    assert result.source_type is SourceType.NONE
    assert not result.available
    assert (result.confidence_level, result.confidence_stars, result.text) == (0, "", "")
