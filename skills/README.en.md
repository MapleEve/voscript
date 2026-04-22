<sub>🌐 [中文](README.md) · **English**</sub>

# VoScript Skills

**AI agent skill package for the VoScript speech transcription service.**

This repository provides loadable skills for any AI agent to drive [VoScript](https://github.com/MapleEve/VoScript), covering the full workflow: audio submission, job polling, transcript retrieval, subtitle export, and voiceprint management.

## Features

- **Agent-agnostic**: works in Claude, Codex, Trae, Hermes, OpenClaw, and any other AI agent — no vendor-specific features required.
- **Self-hosted first**: talks to a local or privately hosted VoScript service. No cloud account needed.
- **Pure Python + requests**: scripts depend only on the standard library and `requests` — no extra install overhead.
- **Bilingual docs**: SKILL.md and reference docs ship in Chinese for native readers; this README covers English users.

## Directory Structure

```
voscript-api/
  SKILL.md                   # Main skill doc (all workflows)
  .meta.json                 # Skill metadata
  references/
    configuration.md         # Service URL + API Key configuration
    job-lifecycle.md         # Job state machine and lifecycle
    voiceprint-guide.md      # Voiceprint enrollment + AS-norm scoring guide
    export-formats.md        # SRT / TXT / JSON format spec
  scripts/
    common.py                # VoScriptClient base client
    submit_audio.py          # Submit audio for transcription
    poll_job.py              # Poll job status
    fetch_result.py          # Fetch complete transcription result
    export_transcript.py     # Export subtitles (SRT / TXT / JSON)
    list_transcriptions.py   # List all transcription records
    enroll_voiceprint.py     # Enroll a voiceprint
    list_voiceprints.py      # List enrolled voiceprints
    assign_speaker.py        # Manually assign a segment's speaker
    manage_voiceprint.py     # View / rename / delete voiceprints
    rebuild_cohort.py        # Rebuild AS-norm cohort
```

## Installation

### Option 1: npx (recommended)

```bash
npx voscript-skills
```

Installs to `~/.claude/skills/voscript-api/` by default.

**Install for all detected agents:**

```bash
npx voscript-skills --all
```

**Install for a specific agent:**

```bash
npx voscript-skills --agent trae
npx voscript-skills --dir /custom/skills/path
```

**Uninstall:**

```bash
npx voscript-skills --uninstall
```

### Option 2: Git clone (manual)

```bash
git clone https://github.com/MapleEve/voscript-skills.git ~/.claude/skills/
```

After cloning, `~/.claude/skills/voscript-api/` is the skill directory.

### Known Agent Directories

| Agent | Skills directory |
| --- | --- |
| Claude Code | `~/.claude/skills/` |
| Trae | `~/.trae/context/skills/` |
| Other | use `--dir` flag |

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

### Example

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

## Loading in an AI Agent

Clone this repo into your agent's skills directory, then load `voscript-api/SKILL.md`. The agent will pick up the workflow definitions and scripts automatically.

## Issues & Feedback

Issues and Discussions are disabled on this repo. Please file on the main project:

→ **[github.com/MapleEve/VoScript/issues](https://github.com/MapleEve/VoScript/issues)**

## License

Same as the main VoScript project. See [LICENSE](LICENSE).
