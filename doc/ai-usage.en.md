# Integration Guide for AI Agents

[简体中文](./ai-usage.zh.md) | **English**

> This document is written for **AI agents / LLMs**. For humans see
> [`quickstart.en.md`](./quickstart.en.md) and [`api.en.md`](./api.en.md).
>
> If you are an AI, read the whole thing before calling any endpoint.

## Who you are, what this service is

You are an AI agent that needs to turn audio into timestamped text labelled
with speaker names. This service (`voscript`) is the
**stateful backend** that does the heavy lifting. It:

1. Accepts audio → runs whisper transcription + pyannote diarization +
   WeSpeaker ResNet34 speaker embedding extraction
2. Maintains a **persistent voiceprint library** so that a returning
   speaker's `SPEAKER_XX` label is upgraded to a real name automatically
3. Lets you **enroll** a `SPEAKER_XX` as a specific person after a job
   completes, so that from then on that person is auto-recognized

## Critical facts (memorize these)

1. **Processing is async.** `POST /api/transcribe` returns only a job id.
   You **must** then poll `/api/jobs/{id}` until
   `status == "completed"` or `"failed"`. Poll, don't sleep-then-assume.
2. **Even short audio can take tens of seconds.** First boot adds model
   load time (can be 2+ minutes). Don't give up after a single poll.
3. **Auth is via header, not query string**:
   ```
   Authorization: Bearer <API_KEY>
   or
   X-API-Key: <API_KEY>
   ```
4. **Enroll with `speaker_label` (the raw `SPEAKER_XX`), NOT `speaker_name`
   (the display name).** This is the #1 trap. When the service auto-matches
   a known voiceprint, `speaker_name` becomes e.g. "Alice" but
   `speaker_label` stays `SPEAKER_00`. Passing `speaker_name` to enroll
   will 404.
5. **The voiceprint match threshold is adaptive.** The base threshold is
   `VOICEPRINT_THRESHOLD` (default 0.75), but each speaker's effective threshold
   is automatically relaxed based on the cosine variance of their enrolled samples:
   a single-sample speaker gets an effective threshold of ~0.70; higher variance
   relaxes it further; absolute floor is 0.60. In any mode, a non-null `speaker_id`
   means the match cleared the threshold. `similarity` is therefore **not** a fixed
   ">= 0.75 means matched" field.

   AS-norm cohort lifecycle (important):
   - **Fresh install (zero transcriptions)**: cohort size = 0, AS-norm is inactive;
     `identify` runs raw cosine + 0.75 base threshold + per-speaker adaptive relaxation.
   - **cohort size < 10**: `ASNormScorer.score()` returns raw cosine rather than a
     true AS-norm z-score (fallback path); threshold behavior is identical to raw
     cosine mode.
   - **cohort size ≥ 10**: true AS-norm is active; the effective normalized score
     threshold is sample-count-aware around the 0.5 operating point. One-sample
     candidates are deliberately stricter (at least 0.60 by default), stable
     multi-sample candidates stay near the base, and ambiguous top-1/top-2
     scores are left unnamed for review.
   - **Startup path**: if `data/transcriptions/asnorm_cohort.npy` exists, startup
     loads it directly. Otherwise startup scans persisted transcriptions /
     `emb_*.npy` files, rebuilds the cohort, and saves it there.
   - **Refresh timing**: enroll / update operations advance a generation counter.
     A background daemon thread (`cohort-rebuild`) wakes every 60 s and triggers
     an automatic rebuild once the latest enrollment is at least 30 s old. No
     manual action is needed; new embeddings typically enter the matching path
     within about 30-90 s of enrollment. They enter full AS-norm scoring only
     when cohort size is at least 10; otherwise raw-cosine fallback remains
     active. `POST /api/voiceprints/rebuild-cohort` remains available for an
     immediate forced rebuild.
6. **Omitting `language` enables auto-detection.** Whisper detects the language on
   its own; the service also injects an `initial_prompt` that nudges the decoder
   toward Simplified Chinese output (useful for Mandarin audio). The result's
   `params.language` will show `"auto"` instead of a language code. Passing
   `language=zh` or `language=en` explicitly behaves exactly as before.
7. **Submitting the same file twice hits deduplication.** The server computes a
   SHA256 of every upload. There are two dedup outcomes:
   - completed historical hit → `{"id": "...", "status": "completed", "deduplicated": true}`
   - concurrent in-flight hit → `{"id": "...", "status": "queued", "deduplicated": true}`
   In both cases, your request did not start a new worker. Use the returned `id`
   normally — no special handling required beyond polling it.
8. **`POST /api/voiceprints/enroll` treats `speaker_id` as an explicit update
   target, not as a required create-vs-update switch.** If the supplied
   `speaker_id` exists, that exact voiceprint is updated. If it is omitted, or is
   well-formed but not found, the endpoint takes the create path, which may still
   merge into an existing same-name record via name deduplication.

## Recommended flow

```
[you have audio]
     │
     ▼
POST /api/transcribe                (get job_id)
     │
     ▼
GET /api/jobs/{job_id}              (every 2-5 seconds, until completed/failed)
     │
     ▼
parse result.segments
     │
     ├── user just labelled a SPEAKER_XX as a real name?
     │       └── POST /api/voiceprints/enroll  (create or update)
     │
     └── need to hand output to something downstream?
             └── GET /api/export/{tr_id}?format=srt|txt|json
```

## Pseudocode template

```python
import time, requests

BASE = "http://host:8780"
KEY = "<your API key>"
H = {"Authorization": f"Bearer {KEY}"}

# 1. submit
with open("meeting.wav", "rb") as f:
    job = requests.post(
        f"{BASE}/api/transcribe",
        headers=H,
        files={"file": f},
        data={
            # "language": "en",  # optional; omit to auto-detect (Mandarin audio → Simplified Chinese)
            "max_speakers": "4",
            # optional: "denoise_model": "deepfilternet",
        },
    ).json()

job_id = job["id"]

# 2. poll
while True:
    r = requests.get(f"{BASE}/api/jobs/{job_id}", headers=H).json()
    if r["status"] == "completed":
        result = r["result"]
        break
    if r["status"] == "failed":
        raise RuntimeError(r.get("error", "unknown"))
    time.sleep(3)

# 3. consume
for seg in result["segments"]:
    # display with speaker_name, enroll with speaker_label
    print(f"[{seg['start']:.1f}s] {seg['speaker_name']}: {seg['text']}")

# 4. enroll (if user told you SPEAKER_00 is Alice)
requests.post(
    f"{BASE}/api/voiceprints/enroll",
    headers=H,
    data={
        "tr_id": result["id"],
        "speaker_label": "SPEAKER_00",  # raw label, not display name!
        "speaker_name": "Alice",
    },
).raise_for_status()
```

## When to enroll

**Only when the user has told you who a specific `SPEAKER_XX` is.** Don't
guess. Valid triggers:
- User says things like "SPEAKER_00 is Alice" or "the first speaker is Bob"
- User corrects a misattributed segment ("that was actually Carol")
- User clicked an "enroll" button in their UI

**Don't** enroll preemptively, and **don't** treat an existing
`speaker_name` as user-confirmed — it might just be the result of a prior
auto-match.

## Common failures

| Symptom | Meaning | Fix |
| --- | --- | --- |
| `401 Unauthorized` | missing or wrong key | check `Authorization` header |
| `404 Embedding not found for this speaker label` | enroll used the wrong `speaker_label` (passed the display name) | use the raw `SPEAKER_XX` |
| `deduplicated: true` with `status: "queued"` | you hit an in-flight duplicate; another request already owns the job | poll the returned id normally |
| polls stay on `transcribing` forever | long audio or cold model load | keep polling (cap at ~20 min) |
| `status = failed, error = "..."` | exception inside the container | surface `error` to the user, check `docker logs` if needed |
| `503 Failed to persist job state...` / `503 Failed to start background transcription...` | the service could not durably bootstrap the job | retry later; no worker was started for your request |
| empty `segments` | silent / too-short / broken audio | ask the user for a different file |

## Don't do this

- ❌ Don't write `HF_TOKEN` or `API_KEY` into code, logs, or prompts
- ❌ Don't expose `:8780` to untrusted callers
- ❌ Don't directly edit `data/voiceprints/voiceprints.db` — use the API's
  delete / rename instead
- ❌ Don't submit the same audio many times — each submission re-runs
  whisper, wasting GPU
- ❌ Don't confuse `speaker_id` and `speaker_label`:
  - `speaker_label` = `SPEAKER_00`, local to a single recording
  - `speaker_id` = `spk_xxxx`, global voiceprint-library id
- ❌ Don't re-submit the same audio file expecting a fresh transcription — the server's SHA256 deduplication will return either a cached completed result or an existing queued in-flight job (`deduplicated: true`) without re-running Whisper. If a fresh re-transcription is truly needed, first delete the existing transcription via `DELETE /api/transcriptions/{id}`, then re-submit.

## Tips

- Handling multiple audio files? Submit all of them first, collect the
  job ids, then poll in parallel. Don't serialize.
- Need to inject the transcript into a downstream prompt? Use
  `GET /api/export/{tr_id}?format=txt` — it already groups segments by
  speaker on plain text lines.
- If a speaker has been enrolled before, they still show up with
  `speaker_label = SPEAKER_XX` in a new recording, but `speaker_name` will
  be the enrolled name. That is not a bug.
- In completed job results, `speaker_map` can legitimately be `{}` when the
  pipeline had no usable speaker embeddings to persist. `unique_speakers` is
  still available and is derived from `segments[].speaker_name`.

## AI Agent Skill Package

If you're integrating VoScript into an AI agent workflow (Claude, Codex, Trae, Hermes,
OpenClaw, or any other agent), use the official skill package:

**[github.com/MapleEve/voscript-skills](https://github.com/MapleEve/voscript-skills)**

Includes:
- `SKILL.md`: complete documentation for all 11 workflows (configure, submit, poll, export, voiceprint management)
- `scripts/`: 11 ready-to-run Python helper scripts (stdlib + `requests` only)
- `references/`: job state machine, voiceprint guide, AS-norm scoring explanation, export formats

## Related docs

- Full API contract → [`api.en.md`](./api.en.md)
- Voiceprint tuning knobs → [`voiceprint-tuning.en.md`](./voiceprint-tuning.en.md)
- Deployment & troubleshooting → [`quickstart.en.md`](./quickstart.en.md)
- Security considerations → [`security.en.md`](./security.en.md)
- AI agent skill package → [voscript-skills](https://github.com/MapleEve/voscript-skills)
