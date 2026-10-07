import pytest

from yla.analysis.topics import TopicMatcher
from yla.config import TopicConfig

MATCHER = TopicMatcher(
    [
        TopicConfig(name="AI Agent", aliases=["Agent", "Agents", "AI Agents"]),
        TopicConfig(name="RAG", aliases=["Retrieval-Augmented Generation"]),
        TopicConfig(name="LLM"),
        TopicConfig(name="Machine Learning", aliases=["ML"]),
    ]
)


def test_from_terms_matches_names_and_aliases_exactly() -> None:
    assert MATCHER.from_terms(["agent", " RAG ", "Prompt", "Agents"]) == ["AI Agent", "RAG"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Building Reliable AI Agents with LangGraph", ["AI Agent"]),
        ("A gentle intro to Retrieval-Augmented Generation and LLMs", ["RAG"]),  # "LLMs" != "LLM"
        ("What is an LLM? ML basics", ["LLM", "Machine Learning"]),
        ("HTML and XML parsing", []),  # "ML" only as a whole word
        ("agent-based simulation", []),  # hyphenated compounds are not the topic
    ],
)
def test_from_text_matches_whole_words(text: str, expected: list[str]) -> None:
    assert MATCHER.from_text(text) == expected


def test_no_topics() -> None:
    empty = TopicMatcher([])
    assert (empty.from_text("anything"), empty.from_terms(["x"])) == ([], [])


CHINESE = TopicMatcher(
    [
        TopicConfig(name="AI Agent", aliases=["Agent", "AI 代理", "智能體"]),
        TopicConfig(name="LLM", aliases=["大型語言模型"]),
        TopicConfig(name="LangChain"),
        TopicConfig(name="Machine Learning", aliases=["ML", "機器學習"]),
    ]
)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("【生成式AI】什麼是大型語言模型？", ["LLM"]),
        ("機器學習2026：AI 代理入門", ["Machine Learning", "AI Agent"]),
        ("用LangChain打造智能體", ["LangChain", "AI Agent"]),
        ("HTML 與 XML 解析", []),  # ASCII boundaries still apply
    ],
)
def test_chinese_titles_match_without_spaces(text: str, expected: list[str]) -> None:
    assert CHINESE.from_text(text) == expected
