"""PoC: verify whether YouTube transcripts can be fetched from the current network.

Flow per channel:
  1. Resolve @handle -> channel_id (one public page fetch, PoC only)
  2. Read channel RSS feed for the latest videos
  3. Try youtube-transcript-api; classify result as manual / generated / none / blocked / error

Usage:
  python poc/transcript_poc.py                 # default channels, 3 videos each
  python poc/transcript_poc.py --per-channel 5 --handles LangChain StatQuest
"""

from __future__ import annotations

import argparse
import json
import re
import socket
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import feedparser
import httpx
from youtube_transcript_api import YouTubeTranscriptApi

DEFAULT_HANDLES = ["DeepLearningAI", "LangChain", "AssemblyAI", "AndrejKarpathy", "statquest"]
PREFERRED_LANGS = ["en", "en-US", "en-GB", "zh-TW", "zh-Hant", "zh", "zh-Hans"]
RSS_URL = "https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
HEADERS = {"User-Agent": "Mozilla/5.0 (YouTube-Learning-Assistant PoC)", "Accept-Language": "en-US,en;q=0.9"}
CHANNEL_ID_PATTERNS = [
    re.compile(r'<link rel="canonical" href="https://www\.youtube\.com/channel/(UC[\w-]{22})"'),
    re.compile(r'"externalId":"(UC[\w-]{22})"'),
    re.compile(r'"channelId":"(UC[\w-]{22})"'),
]


@dataclass
class VideoResult:
    channel: str
    video_id: str
    title: str
    published: str
    description_chars: int
    status: str  # manual | generated | none | blocked | error
    language: str | None = None
    transcript_chars: int = 0
    available_tracks: list[str] | None = None
    error: str | None = None
    elapsed_sec: float = 0.0


def resolve_channel_id(client: httpx.Client, handle: str) -> str:
    resp = client.get(f"https://www.youtube.com/@{handle}", follow_redirects=True)
    resp.raise_for_status()
    for pattern in CHANNEL_ID_PATTERNS:
        if match := pattern.search(resp.text):
            return match.group(1)
    raise ValueError(f"channel_id not found for @{handle}")


def fetch_feed(client: httpx.Client, channel_id: str) -> feedparser.FeedParserDict:
    resp = client.get(RSS_URL.format(channel_id=channel_id))
    resp.raise_for_status()
    return feedparser.parse(resp.text)


def try_transcript(ytt: YouTubeTranscriptApi, video_id: str) -> dict:
    """Return status/language/chars. Manual captions are preferred over auto-generated."""
    try:
        transcript_list = ytt.list(video_id)
    except Exception as exc:  # noqa: BLE001 - PoC wants to record every failure type
        name = type(exc).__name__
        status = (
            "blocked"
            if name in {"RequestBlocked", "IpBlocked"}
            else ("none" if name in {"TranscriptsDisabled", "NoTranscriptFound"} else "error")
        )
        return {"status": status, "error": f"{name}: {str(exc).splitlines()[0][:200]}"}

    tracks = list(transcript_list)
    track_labels = [f"{t.language_code}{'(auto)' if t.is_generated else ''}" for t in tracks]
    if not tracks:
        return {"status": "none", "available_tracks": track_labels}

    def rank(t) -> tuple[int, int]:
        lang_rank = PREFERRED_LANGS.index(t.language_code) if t.language_code in PREFERRED_LANGS else 99
        return (1 if t.is_generated else 0, lang_rank)

    chosen = sorted(tracks, key=rank)[0]
    try:
        fetched = chosen.fetch()
    except Exception as exc:  # noqa: BLE001
        name = type(exc).__name__
        status = "blocked" if name in {"RequestBlocked", "IpBlocked"} else "error"
        error = f"{name}: {str(exc).splitlines()[0][:200]}"
        return {"status": status, "available_tracks": track_labels, "error": error}

    text = " ".join(snippet.text for snippet in fetched)
    return {
        "status": "generated" if chosen.is_generated else "manual",
        "language": chosen.language_code,
        "transcript_chars": len(text),
        "available_tracks": track_labels,
    }


def run(handles: list[str], per_channel: int, delay: float) -> dict:
    ytt = YouTubeTranscriptApi()
    results: list[VideoResult] = []
    channel_errors: dict[str, str] = {}

    with httpx.Client(headers=HEADERS, timeout=20) as client:
        for handle in handles:
            try:
                channel_id = resolve_channel_id(client, handle)
                feed = fetch_feed(client, channel_id)
            except Exception as exc:  # noqa: BLE001
                channel_errors[handle] = f"{type(exc).__name__}: {exc}"
                print(f"[channel] @{handle}: FAILED {channel_errors[handle]}")
                continue

            print(f"[channel] @{handle} -> {channel_id}, {len(feed.entries)} videos in RSS")
            for entry in feed.entries[:per_channel]:
                started = time.monotonic()
                outcome = try_transcript(ytt, entry.yt_videoid)
                result = VideoResult(
                    channel=handle,
                    video_id=entry.yt_videoid,
                    title=entry.title,
                    published=entry.published,
                    description_chars=len(entry.get("summary", "")),
                    elapsed_sec=round(time.monotonic() - started, 2),
                    **outcome,
                )
                results.append(result)
                lang = result.language or "-"
                chars = result.transcript_chars
                print(f"  - {result.status:<9} {lang:<6} {chars:>7} chars | {result.title[:60]}")
                if result.error:
                    print(f"      {result.error}")
                time.sleep(delay)

    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return {
        "run_at": datetime.now(UTC).isoformat(),
        "host": socket.gethostname(),
        "summary": counts,
        "channel_errors": channel_errors,
        "videos": [asdict(r) for r in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--handles", nargs="+", default=DEFAULT_HANDLES)
    parser.add_argument("--per-channel", type=int, default=3)
    parser.add_argument("--delay", type=float, default=1.5, help="seconds between transcript requests")
    parser.add_argument("--label", default="local", help="tag for the output file, e.g. local / modal")
    args = parser.parse_args()

    report = run(args.handles, args.per_channel, args.delay)
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{args.label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary: {report['summary']}")
    print(f"Report saved to {out_file}")


if __name__ == "__main__":
    main()
