# voscript

[简体中文](./README.md) | **English**

Self-hosted GPU transcription service — **meeting transcripts that remember
their speakers.** A small HTTP API that turns audio into timestamped text
labelled with speaker names, and that auto-recognizes returning voices
across recordings.

```
Audio  ──►  faster-whisper large-v3  (transcription)
        ──►  pyannote 3.1             (speaker diarization)
        ──►  ECAPA-TDNN               (speaker embeddings)
        ──►  VoiceprintDB             (cosine match vs. enrolled speakers)
        ──►  timestamped text with identified speaker names
```

What sets it apart from a plain whisper wrapper: the **persistent
voiceprint library**. Enroll a speaker once, and from then on every
recording they appear in gets their real name automatically.

> Example consumer: [OpenPlaud(Maple)](https://github.com/MapleEve/openplaud)
> uses voscript as the backend for meeting recordings. voscript itself is
> just an HTTP service — any client that can POST multipart audio works.

## Documentation

All detailed docs live in [`doc/`](./doc/). Chinese is the default, every
page has an English counterpart:

| Topic | 中文 | English |
| --- | --- | --- |
| Quickstart | [quickstart.zh.md](./doc/quickstart.zh.md) | [quickstart.en.md](./doc/quickstart.en.md) |
| API reference | [api.zh.md](./doc/api.zh.md) | [api.en.md](./doc/api.en.md) |
| **Install guide for AI agents** | [ai-install.zh.md](./doc/ai-install.zh.md) | [ai-install.en.md](./doc/ai-install.en.md) |
| **Usage guide for AI agents** | [ai-usage.zh.md](./doc/ai-usage.zh.md) | [ai-usage.en.md](./doc/ai-usage.en.md) |
| Security policy | [security.zh.md](./doc/security.zh.md) | [security.en.md](./doc/security.en.md) |
| Benchmarks (real-audio wall clock + resource usage) | [benchmarks.zh.md](./doc/benchmarks.zh.md) | [benchmarks.en.md](./doc/benchmarks.en.md) |
| Changelog | [changelog.zh.md](./doc/changelog.zh.md) | [changelog.en.md](./doc/changelog.en.md) |

First-time deployers: start with the [Quickstart](./doc/quickstart.en.md).
AI agents integrating the API: read the [AI usage guide](./doc/ai-usage.en.md).
AI agents deploying the service for a user: read the
[AI install guide](./doc/ai-install.en.md).

## Features

- **Async job pipeline**: `queued → converting → transcribing → identifying → completed`
- **Chinese + multilingual transcription** (WhisperX + faster-whisper large-v3, **word-level timestamps** via forced alignment)
- **Speaker diarization** (pyannote 3.1) + **WeSpeaker ResNet34** embeddings
- **Persistent voiceprints**: enroll once, auto-match across future recordings (cosine similarity ≥ 0.75). sqlite + sqlite-vec under the hood — top-k nearest-neighbour search scales to thousands of speakers
- **Stable HTTP contract**: `/api/transcribe`, `/api/jobs/{id}`, `/api/voiceprints*`, etc. — any HTTP client works
- **Container runs as non-root**; all `/api/*` routes accept optional Bearer / `X-API-Key` auth (constant-time compare); uploads capped by `MAX_UPLOAD_BYTES`; voiceprint DB is concurrency-safe with atomic writes — full hardening list in [`doc/security.en.md`](./doc/security.en.md)
- Minimal built-in web UI at `/` for manual testing

## 30-second start

```bash
git clone https://github.com/MapleEve/voscript.git
cd voscript

cp .env.example .env
# edit .env — at minimum set HF_TOKEN and API_KEY

docker compose up -d --build
curl -sf http://localhost:8780/healthz
```

Full steps + troubleshooting in [`doc/quickstart.en.md`](./doc/quickstart.en.md).

## How to integrate

voscript is a plain HTTP service — no specific client is required. Anything
that can send `multipart/form-data` works (curl, axios, requests, browser
uploads, …).

A typical integration — OpenPlaud(Maple), under Settings → Transcription:

- **Private transcription base URL**: `http://<host>:8780`
- **Private transcription API key**: the same `API_KEY` as in `.env`

After that its worker routes every recording through this service. If
you're writing your own client, the full contract + error table lives in
[`doc/api.en.md`](./doc/api.en.md).

## License

MIT — see [LICENSE](./LICENSE).
