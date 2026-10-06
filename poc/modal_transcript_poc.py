"""Run the same transcript PoC from Modal's cloud IP.

Usage (after `pip install modal` and `modal setup`):
  modal run poc/modal_transcript_poc.py
"""

import json
from pathlib import Path

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(Path(__file__).parent / "requirements.txt"))
    .add_local_python_source("transcript_poc")
)
app = modal.App("yla-transcript-poc", image=image)


@app.function(timeout=600)
def run_remote(per_channel: int = 3) -> dict:
    import transcript_poc

    return transcript_poc.run(transcript_poc.DEFAULT_HANDLES, per_channel, delay=1.5)


@app.local_entrypoint()
def main(per_channel: int = 3) -> None:
    report = run_remote.remote(per_channel)
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / "modal_latest.json"
    out_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Summary: {report['summary']}")
    print(f"Report saved to {out_file}")
