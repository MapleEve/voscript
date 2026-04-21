# 🎙️ VoScript

[简体中文](./README.md) | **English**

<p align="center">
  <a href="https://github.com/MapleEve/voscript/actions/workflows/ci.yml">
    <img src="https://img.shields.io/github/actions/workflow/status/MapleEve/voscript/ci.yml?branch=main&style=for-the-badge" alt="CI" />
  </a>
  <img src="https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge" alt="License: MIT" />
  <img src="https://img.shields.io/badge/Docker-ready-blue?style=for-the-badge&logo=docker" alt="Docker ready" />
</p>

**Self-hosted GPU transcription service — meeting transcripts that remember their speakers.**

A small HTTP API that turns audio into timestamped text labelled with speaker names, and that auto-recognizes returning voices across recordings.

```
Audio  ──►  faster-whisper large-v3              (transcription)
        ──►  pyannote 3.1                         (speaker diarization)
        ──►  DeepFilterNet / noisereduce (optional denoising)
        ──►  WeSpeaker ResNet34                   (speaker embeddings)
        ──►  VoiceprintDB                         (cosine match vs. enrolled speakers)
        ──►  timestamped text with identified speaker names
```

[Quickstart](#30-second-start) · [API Reference](./doc/api.en.md) · [Security](./doc/security.en.md) · [Benchmarks](./doc/benchmarks.en.md) · [Changelog](./doc/changelog.en.md)

## 30-second start

```bash
git clone https://github.com/MapleEve/voscript.git
cd voscript

cp .env.example .env
# edit .env — at minimum set HF_TOKEN and API_KEY

docker compose up -d --build
curl -sf http://localhost:8780/healthz
```

> **Security note**: set a strong `API_KEY` in `.env` before exposing this service on any network. All `/api/*` routes require Bearer or `X-API-Key` authentication when `API_KEY` is configured.

Full steps + troubleshooting in [`doc/quickstart.en.md`](./doc/quickstart.en.md).

## Features

- **Async job pipeline**: `queued → converting → denoising (optional) → transcribing → identifying → completed`
- **Chinese + multilingual transcription** (WhisperX + faster-whisper large-v3, **word-level timestamps** via forced alignment; omit `language` to auto-detect — Mandarin audio outputs Simplified Chinese)
- **Speaker diarization** (pyannote 3.1) + **WeSpeaker ResNet34** embeddings
- **Adaptive voiceprint threshold**: `VOICEPRINT_THRESHOLD` (default 0.75) is the base; the actual threshold relaxes per-speaker based on intra-cluster std of enrolled embeddings — fixed −0.05 for 1 sample, `min(3×std, 0.10)` for 2+, floor at 0.60. Lifted recall from 50% to 70% on 10 real recordings with zero false positives
- **Optional denoising with SNR gate**: `DENOISE_MODEL` (`none` | `deepfilternet` | `noisereduce`); `DENOISE_SNR_THRESHOLD` (default 10.0 dB) — audio above this SNR is considered clean and skipped automatically, preventing DeepFilterNet from degrading already-clean recordings
- **AS-norm voiceprint scoring**: at startup, automatically builds an impostor cohort from existing transcription embeddings and applies Adaptive Score Normalization — eliminates speaker-dependent baseline bias, ~15–30% relative EER improvement
- **Persistent voiceprints**: enroll once, auto-match across future recordings. sqlite + sqlite-vec under the hood — top-k nearest-neighbour search scales to thousands of speakers
- **File hash deduplication**: submitting the same file twice returns the existing result immediately, skipping Whisper GPU inference
- **Stable HTTP contract**: `/api/transcribe`, `/api/jobs/{id}`, `/api/voiceprints*`, etc. — any HTTP client works
- **Container runs as non-root**; all `/api/*` routes accept optional Bearer / `X-API-Key` auth (constant-time compare); uploads capped by `MAX_UPLOAD_BYTES`; voiceprint DB is concurrency-safe with atomic writes — full hardening list in [`doc/security.en.md`](./doc/security.en.md)
- **Job state persistence**: completed transcriptions remain accessible via `GET /api/transcriptions/{id}` after restart; in-flight status written to `status.json`
- **Layered architecture**: `app/` split into `config / api/routers / services / pipeline`; `main.py` is a ~160-line orchestration entry point only
- **Path traversal protection**: `_safe_tr_dir()` + regex-validated `job_id` path params prevent directory traversal attacks
- Minimal built-in web UI at `/` for manual testing

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

## Contributing

PRs welcome — please read [CONTRIBUTING.md](./CONTRIBUTING.md) first.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MapleEve/voscript&type=date)](https://www.star-history.com/#MapleEve/voscript&type=date)

## License

MIT — see [LICENSE](./LICENSE).
