<sub>🌐 <a href="README.md">中文</a> · <b>English</b></sub>

<div align="center">

# VoScript Skills 🧰

> *"You have VoScript. You have an AI agent. They don't talk to each other yet."*

<a href="https://www.npmjs.com/package/voscript-skills">
  <img src="https://img.shields.io/npm/v/voscript-skills?style=flat-square" alt="npm version" />
</a>
<a href="https://www.npmjs.com/package/voscript-skills">
  <img src="https://img.shields.io/npm/dm/voscript-skills?style=flat-square" alt="npm downloads" />
</a>
<a href="https://github.com/MapleEve/VoScript">
  <img src="https://img.shields.io/badge/VoScript-main%20project-blue?style=flat-square" alt="Main project" />
</a>
<a href="./LICENSE">
  <img src="https://img.shields.io/badge/License-Free%20Personal%20%C2%B7%20Commercial%20Ask-orange?style=flat-square" alt="License" />
</a>

<br>

One command to install. Any AI agent (Claude Code, Trae, Cursor…) immediately knows how to drive VoScript.<br>
Full workflow coverage: submit audio, poll jobs, export subtitles, manage voiceprints.<br>
Pure Python + requests. Zero extra dependencies.

<br>

[Install](#installation) · [Quick Start](#quick-start) · [Supported Agents](#supported-agents) · [Main project](https://github.com/MapleEve/VoScript)

</div>

---

## Sound familiar?

> You deployed VoScript. You're using an AI agent to automate tasks. But to make the agent submit a recording, wait for results, and export subtitles, you'd need to write a pile of glue code first.

> Or: the agent hit the API, but didn't know which endpoint to call, when the job would finish, or why `speaker_label` and the display name are different things.

This skill package fixes that. **Load once, and the agent knows the full VoScript workflow.**

---

## Installation

### Option 1: npx (recommended)

```bash
npx voscript-skills
```

Installs to `~/.claude/skills/voscript-api/` by default. Claude Code picks it up automatically.

**Install for all detected agents at once:**

```bash
npx voscript-skills --all
```

**Specify an agent or directory:**

```bash
npx voscript-skills --agent trae      # claude / trae / cursor / codex / gemini / hermes / openclaw / windsurf / cline / roo
npx voscript-skills --dir /custom/skills/path
```

**Uninstall:**

```bash
npx voscript-skills --uninstall
```

### Option 2: Git clone

```bash
git clone https://github.com/MapleEve/voscript-skills.git ~/.claude/skills/
```

After cloning, `~/.claude/skills/voscript-api/` is the skill directory.

---

## Supported Agents

| Agent | --agent | Skills directory | Auto-detected |
| --- | --- | --- | --- |
| Claude Code | `claude` | `~/.claude/skills/` | ✅ |
| Trae | `trae` | `~/.trae/context/skills/` | ✅ |
| Cursor | `cursor` | `~/.cursor/rules/skills/` | ✅ |
| OpenAI Codex CLI | `codex` | `~/.codex/skills/` | ✅ |
| Gemini CLI | `gemini` | `~/.gemini/skills/` | ✅ |
| Hermes | `hermes` | `~/.hermes/skills/` | ✅ |
| OpenClaw | `openclaw` | `~/.openclaw/skills/` | ✅ |
| Windsurf | `windsurf` | `~/.codeium/windsurf/skills/` | ✅ |
| Cline | `cline` | `~/.cline/skills/` | ✅ |
| Roo Code | `roo` | `~/.roo/skills/` | ✅ |
| Other | — | use `--dir` flag | — |

---

## What you get

**11 ready-to-run scripts covering the full workflow**

| Script | Purpose |
| --- | --- |
| `submit_audio.py` | Submit an audio file, get a job ID |
| `poll_job.py` | Progress-bar polling, waits until complete |
| `fetch_result.py` | Fetch complete transcript (with speaker names) |
| `export_transcript.py` | Export SRT / TXT / JSON |
| `list_transcriptions.py` | List all transcription records |
| `enroll_voiceprint.py` | Enroll a voiceprint (with speaker_label guard) |
| `list_voiceprints.py` | List enrolled voiceprints |
| `assign_speaker.py` | Manually assign a segment's speaker |
| `manage_voiceprint.py` | View / rename / delete voiceprints |
| `rebuild_cohort.py` | Rebuild AS-norm cohort |
| `common.py` | Base client + diagnostic helpers |

Every script outputs a structured failure report on error — so the agent knows exactly what went wrong and how to fix it.

---

## Quick Start

### Prerequisites

- Python 3.8+
- `pip install requests`
- A running [VoScript server](https://github.com/MapleEve/VoScript)

### Configuration

```bash
export VOSCRIPT_URL=http://localhost:7880
export VOSCRIPT_API_KEY=your_api_key_here
```

Or pass `--url` / `--api-key` directly to each script.

### Examples

```bash
# Submit audio
python voscript-api/scripts/submit_audio.py --file meeting.mp3

# Poll until complete
python voscript-api/scripts/poll_job.py --job-id tr_xxxxxxxx

# Get result
python voscript-api/scripts/fetch_result.py --tr-id tr_xxxxxxxx

# Export SRT
python voscript-api/scripts/export_transcript.py --tr-id tr_xxxxxxxx --format srt

# Enroll voiceprint (use speaker_label, NOT display name!)
python voscript-api/scripts/enroll_voiceprint.py \
  --tr-id tr_xxxxxxxx \
  --speaker-label SPEAKER_00 \
  --speaker-name "John"
```

---

## Loading in an AI Agent

Install to your agent's skills directory, then load `voscript-api/SKILL.md`. The agent reads the workflow definitions and script usage automatically — no extra configuration needed.

---

## Issues & Feedback

Issues and Discussions are disabled on this repo. File them on the main project:

→ **[github.com/MapleEve/VoScript/issues](https://github.com/MapleEve/VoScript/issues)**

---

## License

Same as the main VoScript project. See [LICENSE](LICENSE).
