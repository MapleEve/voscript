<div align="center">

# 🎙️ VoScript

[简体中文](./README.md) | **English**

<a href="https://github.com/MapleEve/voscript/actions/workflows/ci.yml">
  <img src="https://img.shields.io/github/actions/workflow/status/MapleEve/voscript/ci.yml?branch=main&style=for-the-badge" alt="CI" />
</a>
<a href="https://github.com/MapleEve/voscript/releases">
  <img src="https://img.shields.io/github/v/release/MapleEve/voscript?style=for-the-badge" alt="Release" />
</a>
<a href="https://hub.docker.com/r/mapleeve/voscript">
  <img src="https://img.shields.io/badge/Docker-ready-blue?style=for-the-badge&logo=docker" alt="Docker ready" />
</a>
<a href="./LICENSE">
  <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg?style=for-the-badge" alt="License: Apache 2.0" />
</a>

**Meeting recordings → transcripts with real speaker names. Self-hosted, GPU-powered, remembers every voice.**

[Quickstart](./doc/quickstart.en.md) · [API Reference](./doc/api.en.md) · [Security](./doc/security.en.md) · [Benchmarks](./doc/benchmarks.en.md) · [Changelog](./doc/changelog.en.md)

</div>

---

You have a meeting recording with six people. You want to know who said what. Whisper gives you a wall of text. pyannote can split it into "Speaker A / Speaker B / Speaker C" — but it doesn't know who anyone is. You still have to label every recording by hand.

VoScript fixes that: **enroll a voice once, and it gets automatically identified in every future recording**. Not "Speaker 2" — "Maple".

```
Audio  ──►  faster-whisper large-v3     transcription + word-level timestamps
       ──►  pyannote 3.1                speaker diarization
       ──►  WeSpeaker ResNet34           speaker embeddings
       ──►  VoiceprintDB (AS-norm)       match against enrolled voices
       ──►  timestamped transcript with real speaker names
```

## 30-second start

> **Security**: set a strong `API_KEY` in `.env` before exposing this on any network. Without it, anyone can delete your voiceprint library or trigger GPU jobs.

```bash
git clone https://github.com/MapleEve/voscript.git && cd voscript
cp .env.example .env        # at minimum: HF_TOKEN and API_KEY
docker compose up -d --build
curl -sf http://localhost:8780/healthz
```

Full setup + troubleshooting → [`doc/quickstart.en.md`](./doc/quickstart.en.md)

## Features

- **Persistent voiceprint library** — enroll once, auto-match across all future recordings. sqlite + sqlite-vec under the hood, top-k nearest-neighbour search, scales to thousands of speakers
- **AS-norm scoring** — builds an impostor cohort from existing transcription embeddings at startup; eliminates speaker-dependent baseline bias, ~15–30% relative EER improvement
- **Adaptive threshold** — each speaker's match threshold relaxes dynamically based on enrollment variance; lifted recall from 50% to 70% on 10 real recordings with zero false positives
- **Speaker cluster consolidation** — when diarization splits one person into multiple clusters, they're automatically merged to a single label
- **Word-level timestamps** — WhisperX forced alignment, every word precisely timed
- **Optional denoising with SNR gate** — DeepFilterNet / noisereduce; audio above the SNR threshold is treated as clean and skipped automatically (prevents degrading already-clean recordings)
- **File hash deduplication** — submitting the same file twice returns the existing result immediately, no GPU re-run
- **Job persistence** — completed transcriptions remain accessible after restart
- **Ngram dedup** — `no_repeat_ngram_size` parameter suppresses repetitive filler words in the transcript
- **Plain HTTP contract** — any client that can send multipart/form-data works, no framework lock-in

Security: path traversal protection, non-root container, upload size cap, constant-time auth, atomic writes — full list in [`doc/security.en.md`](./doc/security.en.md)

## Integration

It's a plain HTTP service. Two config values and you're done:

- **Transcription base URL**: `http://<host>:8780`
- **API key**: the `API_KEY` you set in `.env`

[BetterAINote](https://github.com/MapleEve/openplaud) connects this way. Any other client works the same. Full API contract → [`doc/api.en.md`](./doc/api.en.md)

## Documentation

| Topic | 中文 | English |
| --- | --- | --- |
| Quickstart | [quickstart.zh.md](./doc/quickstart.zh.md) | [quickstart.en.md](./doc/quickstart.en.md) |
| API reference | [api.zh.md](./doc/api.zh.md) | [api.en.md](./doc/api.en.md) |
| Install guide for AI agents | [ai-install.zh.md](./doc/ai-install.zh.md) | [ai-install.en.md](./doc/ai-install.en.md) |
| Usage guide for AI agents | [ai-usage.zh.md](./doc/ai-usage.zh.md) | [ai-usage.en.md](./doc/ai-usage.en.md) |
| Security policy | [security.zh.md](./doc/security.zh.md) | [security.en.md](./doc/security.en.md) |
| Benchmarks | [benchmarks.zh.md](./doc/benchmarks.zh.md) | [benchmarks.en.md](./doc/benchmarks.en.md) |
| Changelog | [changelog.zh.md](./doc/changelog.zh.md) | [changelog.en.md](./doc/changelog.en.md) |

## Contributing

PRs welcome — read [CONTRIBUTING.md](./CONTRIBUTING.md) first.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MapleEve/voscript&type=date)](https://www.star-history.com/#MapleEve/voscript&type=date)

## License

Apache 2.0 — [LICENSE](./LICENSE)
