<sub>🌐 <b>English</b> · <a href="README.md">中文</a></sub>

<div align="center">

# VoScript 🎙️

> *"You finished the recording. You want to know who said what — not what 'Speaker A' said."*

<a href="https://github.com/MapleEve/voscript/actions/workflows/ci.yml">
  <img src="https://img.shields.io/github/actions/workflow/status/MapleEve/voscript/ci.yml?branch=main&style=flat-square" alt="CI" />
</a>
<a href="https://github.com/MapleEve/voscript/releases">
  <img src="https://img.shields.io/github/v/release/MapleEve/voscript?style=flat-square" alt="Release" />
</a>
<a href="https://hub.docker.com/r/mapleeve/voscript">
  <img src="https://img.shields.io/badge/Docker-ready-blue?style=flat-square&logo=docker" alt="Docker" />
</a>
<a href="./LICENSE">
  <img src="https://img.shields.io/badge/License-Free%20Personal%20%C2%B7%20Commercial%20Ask-orange?style=flat-square" alt="License" />
</a>

<br>

After the meeting, you want to know who said what — without manually replaying the recording.<br>
Enroll a voice once, recognized automatically from then on. Your data stays on your server, not a cloud.<br>
Full HTTP API — plug into any workflow or AI agent pipeline.

<br>

[Quickstart](./doc/quickstart.en.md) · [API Reference](./doc/api.en.md) · [Benchmarks](./doc/benchmarks.en.md) · [Changelog](./doc/changelog.en.md)

</div>

---

## Sound familiar?

> After every meeting, you open the recording and manually tag names — "this part is Maple, this part is Tom..." A 90-minute meeting takes another 45 minutes to label.

> You tried a speaker diarization tool. Got back Speaker A, Speaker B, Speaker C. Still had to figure out who was who.

VoScript fixes that. **Enroll a voice once, and every future recording automatically gets that person's real name** — not "Speaker 2", but "Maple".

---

## Get started

```bash
git clone https://github.com/MapleEve/voscript.git && cd voscript
cp .env.example .env   # fill in HF_TOKEN and API_KEY
docker compose up -d --build
```

Open `http://localhost:8780` in a browser, upload a recording, wait for results.

> Security: set a strong `API_KEY` in `.env` before exposing this on any network. Without it, anyone can modify your voiceprint library or trigger GPU jobs.

Full setup + troubleshooting → [`doc/quickstart.en.md`](./doc/quickstart.en.md)

---

## Two ways to use it

### Built-in web panel — open a browser and start working

No code needed. The panel has two tabs:

- **Transcribe**: upload an audio file, pick your settings, submit, get results
- **Voiceprint library**: enroll speakers (upload sample → name → save), delete, browse

Best for: occasional recordings, one-off transcription tasks, anyone who doesn't want to touch an API.

### API integration — fully automated pipeline

Point your tool at the service URL with an API Key, and recordings flow in, transcripts flow out. [BetterAINote](https://github.com/MapleEve/BetterAINote) connects this way. Any HTTP client works.

Best for: long-term use, teams with shared recordings, existing audio workflows.

---

## What you get

**Transcription output**

- Timestamped transcript with every word precisely aligned
- Real speaker names on every line (unrecognized voices labeled Unknown)
- Handles multilingual recordings including Chinese and English

**Voiceprint system**

- Enroll today — recordings three years from now still match. Database is a plain file you can back up and move
- Submit the same file twice and the second call returns instantly — no GPU re-run
- Noisy recordings are auto-denoised; clean recordings are skipped automatically (prevents degrading good audio)

**How you use it**

- Built-in web panel for uploads, results, and voiceprint management — no code required
- Plain HTTP API for integration — any tool that can send a request works, no framework lock-in

---

## How it works

```
Audio  ──►  faster-whisper large-v3     transcription + word-level timestamps
       ──►  pyannote 3.1                speaker diarization
       ──►  WeSpeaker ResNet34           speaker embeddings
       ──►  VoiceprintDB (AS-norm)       match against enrolled voices
       ──►  timestamped transcript with real speaker names
```

Speaker matching uses AS-norm scoring to eliminate speaker-dependent baseline bias, combined with adaptive thresholds that relax per-speaker based on enrollment variance. Measured on 10 real recordings: recall 50% → 70%, zero false positives.

Full technical details → [`doc/benchmarks.en.md`](./doc/benchmarks.en.md)

---

## Documentation

| Topic | 中文 | English |
| --- | --- | --- |
| Quickstart | [quickstart.zh.md](./doc/quickstart.zh.md) | [quickstart.en.md](./doc/quickstart.en.md) |
| API reference | [api.zh.md](./doc/api.zh.md) | [api.en.md](./doc/api.en.md) |
| Install guide for AI agents | [ai-install.zh.md](./doc/ai-install.zh.md) | [ai-install.en.md](./doc/ai-install.en.md) |
| Usage guide for AI agents | [ai-usage.zh.md](./doc/ai-usage.zh.md) | [ai-usage.en.md](./doc/ai-usage.en.md) |
| AI agent skill package | [voscript-skills](https://github.com/MapleEve/voscript-skills) | [voscript-skills](https://github.com/MapleEve/voscript-skills) |
| Security policy | [security.zh.md](./doc/security.zh.md) | [security.en.md](./doc/security.en.md) |
| Benchmarks | [benchmarks.zh.md](./doc/benchmarks.zh.md) | [benchmarks.en.md](./doc/benchmarks.en.md) |
| Changelog | [changelog.zh.md](./doc/changelog.zh.md) | [changelog.en.md](./doc/changelog.en.md) |

---

## Contact

WeChat Official Account: **等枫再来** (Follow on WeChat)

Questions, ideas, or just want to commiserate about voice transcription — come find me.

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MapleEve/voscript&type=date)](https://www.star-history.com/#MapleEve/voscript&type=date)

---

## Contributing & License

PRs welcome — read [CONTRIBUTING.md](./CONTRIBUTING.md) first.

Free for individuals, ask first for business use — [LICENSE](./LICENSE)
