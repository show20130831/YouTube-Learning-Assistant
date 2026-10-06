# YouTube Learning Assistant

> Work in progress.

Daily AI-generated study digests of new videos from the YouTube channels you follow, delivered to LINE.

Every day the assistant checks your channels for new videos, pulls the best available text (manual captions → auto captions → title and description), summarizes it with a free long-context LLM via OpenRouter, and pushes one digest to LINE. Each summary is labeled with a ⭐–⭐⭐⭐ confidence rating based on its source, and the assistant never invents a summary when there isn't enough text.

## Development

```bash
uv sync
cp config/settings.example.yaml config/settings.yaml
cp .env.example .env
uv run yla validate-config
uv run pytest
```

## License

MIT
