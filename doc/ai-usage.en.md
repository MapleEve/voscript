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
   ECAPA-TDNN speaker embedding extraction
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
5. **`similarity >= 0.75` means matched.** Below that, `speaker_id` is
   `null` and `speaker_name` falls back to the raw label.

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
        data={"language": "en", "max_speakers": "4"},
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
| polls stay on `transcribing` forever | long audio or cold model load | keep polling (cap at ~20 min) |
| `status = failed, error = "..."` | exception inside the container | surface `error` to the user, check `docker logs` if needed |
| empty `segments` | silent / too-short / broken audio | ask the user for a different file |

## Don't do this

- ❌ Don't write `HF_TOKEN` or `API_KEY` into code, logs, or prompts
- ❌ Don't expose `:8780` to untrusted callers
- ❌ Don't edit `.npy` files under `data/voiceprints/` — use the API's
  delete / rename instead
- ❌ Don't submit the same audio many times — each submission re-runs
  whisper, wasting GPU
- ❌ Don't confuse `speaker_id` and `speaker_label`:
  - `speaker_label` = `SPEAKER_00`, local to a single recording
  - `speaker_id` = `spk_xxxx`, global voiceprint-library id

## Tips

- Handling multiple audio files? Submit all of them first, collect the
  job ids, then poll in parallel. Don't serialize.
- Need to inject the transcript into a downstream prompt? Use
  `GET /api/export/{tr_id}?format=txt` — it already groups segments by
  speaker on plain text lines.
- If a speaker has been enrolled before, they still show up with
  `speaker_label = SPEAKER_XX` in a new recording, but `speaker_name` will
  be the enrolled name. That is not a bug.

## Related docs

- Full API contract → [`api.en.md`](./api.en.md)
- Deployment & troubleshooting → [`quickstart.en.md`](./quickstart.en.md)
- Security considerations → [`security.en.md`](./security.en.md)
