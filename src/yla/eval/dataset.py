"""The evaluation dataset (``eval/dataset.yaml``) and its local transcript cache.

The dataset file is committed: video IDs, labels with short evidence quotes, and a frozen copy of
the topic list, so labels stay valid when ``settings.yaml`` changes. Transcripts and model outputs
are copyrighted or derived from copyrighted text and stay in git-ignored folders.
"""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from yla.config import TopicConfig
from yla.youtube.transcripts import TranscriptResult, TranscriptStatus

DEFAULT_DATASET_PATH = Path("eval/dataset.yaml")
DEFAULT_CACHE_DIR = Path("eval/cache")
DEFAULT_RUNS_DIR = Path("eval/runs")


class Split(StrEnum):
    DEV = "dev"  # seen while tuning prompts or rules; free to use while developing
    TEST = "test"  # only for decisions


class LabelStatus(StrEnum):
    DRAFT = "draft"  # drafted from the transcript, not yet checked by a person
    CONFIRMED = "confirmed"  # checked by a person against the evidence


class Labels(BaseModel):
    model_config = ConfigDict(extra="forbid")

    topics: list[str]
    # topic -> a short quote from the transcript that shows the video really discusses it
    evidence: dict[str, str] = Field(default_factory=dict)
    status: LabelStatus = LabelStatus.DRAFT
    notes: str | None = None


class Item(BaseModel):
    model_config = ConfigDict(extra="forbid")

    video_id: str
    split: Split
    channel: str
    title: str
    published_at: datetime | None = None
    duration_seconds: int | None = None
    labels: Labels | None = None


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    description: str = ""
    min_description_chars: int = 150
    topics: list[TopicConfig]
    items: list[Item]

    @model_validator(mode="after")
    def _check(self) -> Dataset:
        names = {t.name for t in self.topics}
        seen: set[str] = set()
        for item in self.items:
            if item.video_id in seen:
                raise ValueError(f"duplicate video {item.video_id}")
            seen.add(item.video_id)
            if item.labels:
                unknown = set(item.labels.topics) - names
                if unknown:
                    raise ValueError(f"{item.video_id}: topics not in the frozen list: {sorted(unknown)}")
                if missing := set(item.labels.topics) - set(item.labels.evidence):
                    raise ValueError(f"{item.video_id}: no evidence for {sorted(missing)}")
        return self

    @property
    def labelled(self) -> list[Item]:
        return [i for i in self.items if i.labels is not None]


def load_dataset(path: Path = DEFAULT_DATASET_PATH) -> Dataset:
    return Dataset.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


class CachedVideo(BaseModel):
    """Everything a run needs about one video, kept locally (never committed)."""

    video_id: str
    channel: str
    title: str
    description: str
    transcript_status: TranscriptStatus
    language: str | None = None
    text: str | None = None
    source: str  # "database" (exported read-only) or "youtube" (fetched for evaluation)
    fetched_at: datetime

    def transcript(self) -> TranscriptResult:
        return TranscriptResult(status=self.transcript_status, language=self.language, text=self.text or "")


class Cache:
    def __init__(self, root: Path = DEFAULT_CACHE_DIR) -> None:
        self.root = root

    def _path(self, video_id: str) -> Path:
        return self.root / f"{video_id}.json"

    def get(self, video_id: str) -> CachedVideo | None:
        path = self._path(video_id)
        if not path.exists():
            return None
        return CachedVideo.model_validate_json(path.read_text(encoding="utf-8"))

    def put(self, video: CachedVideo) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self._path(video.video_id).write_text(
            json.dumps(video.model_dump(mode="json"), ensure_ascii=False, indent=1), encoding="utf-8"
        )

    def all(self) -> list[CachedVideo]:
        if not self.root.exists():
            return []
        return [
            CachedVideo.model_validate_json(p.read_text(encoding="utf-8")) for p in self.root.glob("*.json")
        ]
