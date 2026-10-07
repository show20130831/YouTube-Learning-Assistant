from datetime import time
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from yla.cli import app
from yla.config import AppSettings, Secrets, load_settings

EXAMPLE = Path(__file__).parents[1] / "config" / "settings.example.yaml"


def make(**overrides: object) -> AppSettings:
    data: dict[str, object] = {"channels": [{"handle": "LangChain"}], **overrides}
    return AppSettings.model_validate(data)


def test_example_settings_file_is_valid() -> None:
    settings = load_settings(EXAMPLE)
    assert len(settings.enabled_channels) == 5
    assert settings.user.notify_time == time(12, 0)
    assert settings.user.daily_video_limit == 5


def test_missing_file_has_helpful_message(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"settings\.example\.yaml"):
        load_settings(tmp_path / "settings.yaml")


def test_handle_leading_at_is_stripped() -> None:
    assert make(channels=[{"handle": " @statquest "}]).channels[0].handle == "statquest"


def test_duplicate_handles_rejected_case_insensitively() -> None:
    with pytest.raises(ValidationError, match="duplicate channel handles"):
        make(channels=[{"handle": "LangChain"}, {"handle": "@langchain"}])


def test_alias_shared_between_topics_rejected() -> None:
    topics = [{"name": "RAG", "aliases": ["Retrieval"]}, {"name": "Search", "aliases": ["retrieval"]}]
    with pytest.raises(ValidationError, match="used by both"):
        make(topics=topics)


def test_alias_equal_to_own_name_is_allowed() -> None:
    assert make(topics=[{"name": "ML", "aliases": ["ml"]}]).topics[0].name == "ML"


@pytest.mark.parametrize(
    "user",
    [{"timezone": "Mars/Base"}, {"daily_video_limit": 0}, {"daily_video_limit": 11}, {"unknown": 1}],
)
def test_invalid_user_settings_rejected(user: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make(user=user)


def test_at_least_one_channel_required() -> None:
    with pytest.raises(ValidationError):
        AppSettings.model_validate({"channels": []})


def test_secrets_read_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("LINE_USER_ID", "U123")
    secrets = Secrets(_env_file=None)
    assert secrets.openrouter_api_key is not None
    assert secrets.openrouter_api_key.get_secret_value() == "sk-test"
    assert "sk-test" not in repr(secrets)
    assert secrets.line_user_id == "U123"


def test_cli_validate_config_ok() -> None:
    result = CliRunner().invoke(app, ["validate-config", "--path", str(EXAMPLE)])
    assert result.exit_code == 0, result.output
    assert "5 enabled" in result.output


def test_cli_validate_config_invalid(tmp_path: Path) -> None:
    bad = tmp_path / "settings.yaml"
    bad.write_text("channels: []\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["validate-config", "--path", str(bad)])
    assert result.exit_code == 1


def test_channel_id_is_optional_but_validated() -> None:
    assert make(channels=[{"handle": "x"}]).channels[0].channel_id is None
    with pytest.raises(ValidationError):
        make(channels=[{"handle": "x", "channel_id": "not-a-channel-id"}])


def test_duplicate_channel_ids_rejected() -> None:
    cid = "UCC-lyoTfSrcJzA1ab3APAgw"
    with pytest.raises(ValidationError, match="duplicate channel ids"):
        make(channels=[{"handle": "a", "channel_id": cid}, {"handle": "b", "channel_id": cid}])


def test_example_settings_have_channel_ids() -> None:
    assert all(c.channel_id for c in load_settings(EXAMPLE).channels)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "postgresql://u:p@ep-x-pooler.ap-southeast-1.aws.neon.tech/neondb?sslmode=require",
            {"prepare_threshold": None},
        ),
        ("postgresql://u:p@ep-x.ap-southeast-1.aws.neon.tech/neondb?sslmode=require", {}),
        ("postgresql+psycopg://yla:yla@localhost:5432/yla", {}),
    ],
)
def test_prepared_statements_disabled_behind_a_pooler(url: str, expected: dict[str, object]) -> None:
    from yla.db.session import connect_args, normalize_url

    assert connect_args(url) == expected
    assert normalize_url(url).startswith("postgresql+psycopg://")
