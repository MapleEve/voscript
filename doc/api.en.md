# API Reference

[简体中文](./api.zh.md) | **English**

All endpoints live under `http://<host>:8780`. JSON for data, `multipart/form-data`
for file uploads.

For complete env defaults, API override precedence, and ASR / AS-norm internal
defaults that are not public knobs yet, see
[`configuration.en.md`](./configuration.en.md).

## Authentication

With `API_KEY` set, every request except the ones below must carry
`Authorization: Bearer <API_KEY>` **or** `X-API-Key: <API_KEY>`:

| Path | Public | Match |
| --- | --- | --- |
| `GET /` | ✅ bundled web UI | exact |
| `GET /healthz` | ✅ liveness probe | exact |
| `GET /docs` / `/redoc` / `/openapi.json` | ✅ FastAPI auto docs | exact |
| `GET /static/*` | ✅ static assets | `/static/` prefix |
| other `/api/*` | ❌ requires the key | — |

Missing or wrong key → `401 Unauthorized`. Key comparison uses
`hmac.compare_digest` (constant-time). Since 0.2.0, `/docs`, `/redoc`,
`/openapi.json` are **exact-match** public paths — `/docsXYZ` now
returns 401.

## Job lifecycle

```
POST /api/transcribe
    ↓
queued → converting → denoising (if effective denoise_model ≠ none) → transcribing → identifying → completed
                                                                                              ↘ failed
```

The BetterAINote worker polls `/api/jobs/{id}` every 5 seconds and stops
as soon as it sees `completed` or `failed`.

## Endpoints

### `GET /healthz`

```bash
curl http://localhost:8780/healthz
# {"ok":true}
```

### `POST /api/transcribe` — submit a job

Form fields:

| Field | Type | Description |
| --- | --- | --- |
| `file` | file | Required — audio (wav / mp3 / m4a / flac / ogg / webm) |
| `language` | string | Optional, ISO 639-1; omit to auto-detect (Mandarin audio outputs Simplified Chinese) |
| `min_speakers` | int | Optional, `0` = auto |
| `max_speakers` | int | Optional, `0` = auto |
| `denoise_model` | string | Optional. Noise reduction backend: `none`, `deepfilternet`, `noisereduce`. When omitted, the server uses `DENOISE_MODEL` (default `none`). Sending `none` explicitly disables denoising for this request only. |
| `snr_threshold` | float | Optional. SNR gate threshold (dB) for this request only. Audio at or above this level skips denoising. Overrides `DENOISE_SNR_THRESHOLD` (default `10.0`). |
| `no_repeat_ngram_size` | int | Optional, default `0` (disabled). When ≥ 3, suppresses n-gram repetitions in the transcript (e.g. "like like like" → "like"). Values < 3 are treated as `0`. Non-integer values return 422. |

Response (200):

```json
{ "id": "tr_example_id", "status": "queued" }
```

`POST /api/transcribe` has two dedup paths, both keyed by the upload SHA256:

- **Completed-result dedup**: if an identical file already has a completed transcription,
  the endpoint returns that existing job immediately without re-running Whisper:

```json
{ "id": "tr_existing_id", "status": "completed", "deduplicated": true }
```

- **In-flight dedup**: if an identical file is already being processed by another live
  request, the later caller is attached to the first job instead of starting a second
  worker. The response reuses the first job id and stays in `queued` until that job
  advances:

```json
{ "id": "tr_existing_inflight", "status": "queued", "deduplicated": true }
```

In both cases, `deduplicated: true` means **this request did not create a new transcription
worker**. Use the returned `id` normally — poll `/api/jobs/{id}` or export as usual.

**Upload size**: the server streams the upload in chunks and returns
`413` the moment the total exceeds `MAX_UPLOAD_BYTES` (default 2 GiB):

```json
{ "detail": "Upload exceeds MAX_UPLOAD_BYTES (2147483648 bytes)" }
```

The partial file is deleted from `data/uploads/`. Lower the cap in
`.env` if your disk is small (the value is in bytes).

**Filename**: the multipart `filename` is reduced to
`PurePosixPath(filename).name` before use. A client-supplied
`filename=../../etc/passwd.wav` lands on disk as just
`tr_<id>_passwd.wav`.

**503 cases**: `POST /api/transcribe` can also fail before work starts:

- `503 Failed to persist job state — disk error, retry later`
- `503 Failed to start background transcription — retry later`

Example:

```bash
curl -X POST http://localhost:8780/api/transcribe \
     -H "Authorization: Bearer $API_KEY" \
     -F "file=@meeting.wav" \
     -F "language=en" \
     -F "max_speakers=4"
```

Noise reduction precedence is: explicit API field first, then server env. In
practice, omit `denoise_model` to inherit `DENOISE_MODEL`, send
`denoise_model=none` to disable denoising for one request, and send
`snr_threshold` only when this job needs a threshold different from
`DENOISE_SNR_THRESHOLD`.

### `GET /api/jobs/{id}` — poll a job

> **Note**: `GET /api/jobs/{id}` checks the in-memory job dictionary first; on a cache miss it falls back to `data/transcriptions/<id>/status.json` on disk.
> - If a completed job is still present in memory, `result` is served from the in-memory job cache.
> - On a cache miss, completed jobs load `result.json` from disk.
> - If status is in-progress at the time of the miss, it returns `status=failed, error="Process restarted while job was in progress"` (set by `recover_orphan_jobs()` at startup).
> - Returns 404 only if `status.json` does not exist.
>
> **Service restarts no longer leave jobs in an indeterminate state** — clients will always receive a definitive terminal status.

```json
{
  "id": "tr_...",
  "status": "queued | converting | denoising | transcribing | identifying | completed | failed",
  "filename": "meeting.wav",

  "error": "...",     // only when status = failed
  "result": {         // only when status = completed
    "id": "tr_...",
    "language": "en",
    "segments": [
      {
        "id": 0,
        "start": 0.0,
        "end": 4.32,
        "text": "This is the first segment.",
        "speaker_label": "SPEAKER_00",
        "speaker_id": "spk_...",
        "speaker_name": "Alice",
        "similarity": 0.8421,
        "words": [
          { "word": "This", "start": 0.05, "end": 0.18, "score": 0.98 },
          { "word": "is",   "start": 0.18, "end": 0.29, "score": 0.96 }
        ]
      }
    ],
    "speaker_map": {
      "SPEAKER_00": {
        "matched_id": "spk_...",
        "matched_name": "Alice",
        "similarity": 0.8421,
        "embedding_key": "SPEAKER_00"
      }
    },
    "unique_speakers": ["Alice"],
    "params": {
      "language": "en",  // shows "auto" when no language was specified at submit time
      "denoise_model": "none",
      "snr_threshold": 10.0,
      "voiceprint_threshold": 0.75,
      "min_speakers": 0,
      "max_speakers": 0,
      "no_repeat_ngram_size": 0
    },
    "alignment": {
      "status": "succeeded",
      "language": "en",
      "model": null,
      "model_source": "whisperx_default",
      "cache_only": false
    }
  }
}
```

**`speaker_label` is the raw pyannote label** — it never changes even when
an existing voiceprint was matched. Use it as the key for any later
enrollment or rename call.

**Result contract anchors**: completed results report `status="completed"` in
the persisted transcription object. `segments[].speaker_label` is always the
raw diarization cluster label. `segments[].words` and top-level `alignment` are
optional metadata; clients must tolerate either field being absent.

`speaker_id` / `speaker_name`: matching uses an **adaptive threshold**, not a
fixed `0.75` cutoff. Actual logic:

- Base threshold is `VOICEPRINT_THRESHOLD` (default `0.75`).
- Each speaker's effective threshold is relaxed automatically based on the cosine
  spread of their enrolled samples: a one-sample speaker lands around `0.70`;
  higher spread can relax it further (up to `0.10`), with an absolute floor of
  `0.60`.
- Once AS-norm is active (`cohort >= 10`), matching switches to the normalised
  score and uses a sample-count-aware threshold around the `0.5` operating point:
  one-sample speakers are stricter (at least `0.60` by default), stable
  multi-sample speakers stay near the base, and candidates too close to the
  second-best AS-norm score are left unnamed for review.

If the best candidate clears the effective threshold, the service returns the
matched `speaker_id` / `speaker_name`; otherwise `speaker_id` is `null` and
`speaker_name` falls back to the raw label (for example `SPEAKER_00`).

If two diarization labels in the same result resolve to the same display name,
the service keeps both raw `speaker_label` values and disambiguates display
names in segment output, for example `Alice` and `Alice (2)`. Voiceprint naming
does not collapse diarization clusters.

`similarity`: speaker-match score.

- **Raw cosine mode** (`cohort < 10`, including fresh installs): range is `[-1, 1]`
  and usually `[0, 1]`, representing cosine similarity against the enrolled
  speaker average.
- **AS-norm mode** (`cohort >= 10`): this becomes a normalised z-score and is
  therefore unbounded (it can be greater than `1.0` or negative).
- The value is aggregated at the **speaker** level, not per individual segment.
- `speaker_id != null` means the score passed the effective threshold in the
  current mode.

See [`voiceprint-tuning.en.md`](./voiceprint-tuning.en.md) for environment
variables, API parameters, AS-norm `top_n` / cohort / margin defaults, and
tuning guidance.

**`words[]` is a new optional field added in 0.3.0** (WhisperX forced
alignment output). Each entry carries its own `start`/`end`/`score`.
Alignment can be skipped or fail for languages whose align model is unavailable
or disabled; when it does, the key is simply absent from the segment and the job
still finishes. Clients that don't recognize the field should just ignore it.

**`alignment`** records forced-alignment status when available. Common values:
`status=succeeded`, `status=skipped` with `reason=language_disabled`, or
`status=failed` with a sanitized `error_type` and `actionable_hint`. The
default Chinese alignment model is reported as
`jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn`; if an older custom
runtime is blocked by transformers' `torch.load` safety check, `reason` is
`torch_version_blocked` rather than `not_found`. This metadata intentionally
does not expose tokens, hostnames, or local filesystem paths.

**`params`** records the effective settings used for this specific job,
including any per-request overrides. Makes each result self-contained —
no need to cross-reference the original request. See
[`configuration.en.md`](./configuration.en.md) for each setting's source and
default.

Completed `GET /api/jobs/{id}` results and `GET /api/transcriptions/{id}` share the
same payload shape. That means `speaker_map` and `unique_speakers` are available in
the completed job result as well:

- If you need the **latest persisted result after manual segment edits**, prefer
  `GET /api/transcriptions/{id}`. `GET /api/jobs/{id}` may still be serving the
  worker's in-memory completed copy until that cache entry is evicted.
- `speaker_map` may be an empty object when the pipeline produced no usable speaker
  embeddings (for example, all diarized turns were too short to enroll).
- `unique_speakers` is derived from the resolved `segments[].speaker_name` values and
  therefore uses enrolled names when matched, otherwise the raw diarization labels.

### `GET /api/transcriptions` — list past jobs

```json
[
  { "id": "tr_...", "filename": "...", "created_at": "...",
    "segment_count": 42, "speaker_count": 3 }
]
```

### `GET /api/transcriptions/{tr_id}` — full result

Same shape as the completed `result` field inside `GET /api/jobs/{id}`, plus two
aggregation fields for UI / downstream consumers:

| Field | Type | Description |
| --- | --- | --- |
| `speaker_map` | object | `speaker_label → {matched_id, matched_name, similarity, embedding_key}` mapping; reflects the **diarization model's voiceprint match result** and does not change when segments are manually corrected |
| `unique_speakers` | array[string] | Deduplicated list of speaker names, recalculated from the persisted `segments[].speaker_name` values to reflect the latest manual corrections |

### `GET /api/export/{tr_id}`

Query `format=srt | txt | json`. Returns the file as a download.

### Voiceprint library

```
GET    /api/voiceprints
POST   /api/voiceprints/enroll
PUT    /api/voiceprints/{speaker_id}/name
DELETE /api/voiceprints/{speaker_id}
```

#### `GET /api/voiceprints`

```json
[
  { "id": "spk_example_id", "name": "Alice",
    "sample_count": 3,
    "created_at": "2026-04-18T08:06:41.951819",
    "updated_at": "2026-04-18T09:17:02.113207" }
]
```

#### `POST /api/voiceprints/enroll`

> **Note (enroll idempotency)**: `add_speaker` now deduplicates by `name` — re-enrolling a speaker with the same name merges the new embedding into the existing record rather than creating a duplicate.
>
> Pass `speaker_id` only when you intend to update that exact existing voiceprint. If
> the supplied `speaker_id` is well-formed but not found, the endpoint does **not** 404;
> it falls back to the create/name-dedup path.

Form fields:

| Field | Required | Description |
| --- | --- | --- |
| `tr_id` | ✅ | Transcription id, matches `result.id` |
| `speaker_label` | ✅ | **Must** be the raw `SPEAKER_XX` label, not the display name |
| `speaker_name` | ✅ | Display name, e.g. "Alice" |
| `speaker_id` | ❌ | Explicit update target. If this id exists, the endpoint updates that voiceprint and returns `action: "updated"`. If omitted, or if the id is well-formed but not found, the endpoint takes the create path, which may still merge into an existing same-name record via `add_speaker()` dedup. Format must match `^spk_[A-Za-z0-9_-]{1,64}$` (e.g. `spk_example_id`); returns 422 if invalid. |

Response:

```json
{ "action": "created | updated", "speaker_id": "spk_..." }
```

Example:

```bash
curl -X POST http://localhost:8780/api/voiceprints/enroll \
     -H "Authorization: Bearer $API_KEY" \
     -F "tr_id=tr_example_id" \
     -F "speaker_label=SPEAKER_00" \
     -F "speaker_name=Alice"
```

#### `POST /api/voiceprints/rebuild-cohort`

Rebuilds the AS-norm impostor cohort matrix from all existing transcriptions. Manual
rebuilds are still supported, but 0.7.1 also has automatic cohort loading and refresh.

Response:

```json
{ "cohort_size": 313, "skipped": 2, "saved_to": "/data/transcriptions/asnorm_cohort.npy" }
```

`skipped` — number of transcriptions whose embedding files could not be loaded (corrupt or missing `.npy`).

**Cohort lifecycle and behaviour**:

| Cohort size | Identification path | Effective threshold |
| --- | --- | --- |
| 0 (fresh install / no transcriptions) | raw cosine | base 0.75 + adaptive relaxation, floor 0.60 |
| 1–9 (fewer than 10) | raw cosine (`score()` fallback) | same as above |
| ≥ 10 | AS-norm normalised score | ~0.5 (relative to impostor distribution; `VOICEPRINT_THRESHOLD` ignored) |

**Startup behaviour**:

- If `data/transcriptions/asnorm_cohort.npy` already exists, the service loads it
  directly on startup.
- Otherwise it scans persisted transcription results / `emb_*.npy` files and builds
  a fresh cohort, then saves it back to that path.

**Refresh timing**: each enroll / update bumps a generation counter. A background daemon
thread named `cohort-rebuild` wakes every 60 s and calls `maybe_rebuild_cohort()` once
the latest enrollment is at least 30 s old. The rebuild is lock-protected, so the
daemon and `POST /api/voiceprints/rebuild-cohort` cannot run the rebuild concurrently.
**No manual action is needed** — new embeddings usually enter AS-norm scoring within
about 30-90 s of enrollment. Automatic rebuilds protect a larger loaded or persisted
cohort: if the transcription source is empty, has only a few embeddings, or has fewer
embeddings than the current cohort, the daemon keeps the existing `asnorm_cohort.npy`
instead of shrinking it after transcription cleanup. `POST /api/voiceprints/rebuild-cohort`
remains available for an immediate forced rebuild and uses the currently available
embeddings as an explicit manual operation.

#### `PUT /api/voiceprints/{id}/name`

Form `name=<new name>`. Renames only; the embedding is unchanged.

#### `DELETE /api/voiceprints/{id}`

Removes the voiceprint permanently. Future recordings of that person will
not auto-match.

### `PUT /api/transcriptions/{tr_id}/segments/{seg_id}/speaker`

Manually reassign a single segment to a different speaker.

Form fields:

| Field | Required | Description |
| --- | --- | --- |
| `speaker_name` | ✅ | New speaker display name |
| `speaker_id` | ❌ | ID of a registered voiceprint (format: `^spk_[A-Za-z0-9_-]{1,64}$`); omitting this clears any previously assigned `speaker_id` on the segment |

Behavior:

- **Only the targeted segment is updated** — other segments are not affected.
- `speaker_map` **is not modified** — it records the diarization model's voiceprint match result and is not affected by manual corrections.
- `unique_speakers` is recalculated from all segments after each edit to reflect the latest corrections.
- When `speaker_id` is omitted, any stale `speaker_id` on the target segment is explicitly cleared to `null`.

Errors:

- `422` — `speaker_id` format invalid (does not match `^spk_[A-Za-z0-9_-]{1,64}$`)
- `404` — `speaker_id` not found in the voiceprint DB
- `404` — `tr_id` transcription not found
- `404` — `seg_id` not found in this transcription

## Error responses

| Code | Meaning |
| --- | --- |
| 400 | Missing or invalid request field; illegal job_id format (`^tr_[A-Za-z0-9_-]{1,64}$`) / invalid characters in speaker_label / path traversal detected |
| 422 | Field value fails type or value validation; `speaker_id` does not match `^spk_[A-Za-z0-9_-]{1,64}$`; `no_repeat_ngram_size` is not an integer |
| 401 | Missing or wrong API key |
| 404 | Unknown tr_id / speaker_id / missing embedding |
| 413 | Upload exceeded `MAX_UPLOAD_BYTES` (default 2 GiB) — see `/api/transcribe` |
| 503 | Failed to persist initial `queued` status or failed to start the background transcription thread |
| 500 | Server-side exception (check `docker logs voscript`) |
| 504 | ffmpeg transcoding timed out (exceeded `FFMPEG_TIMEOUT_SEC`, default 1800 s) |

Body shape:

```json
{ "detail": "..." }
```

## BetterAINote mapping

| BetterAINote code | Endpoint called |
| --- | --- |
| `submitVoiceTranscribeJob` | `POST /api/transcribe` |
| `pollVoiceTranscribeJob` | `GET /api/jobs/{id}` |
| `VoiceTranscribeClient.listVoiceprints` | `GET /api/voiceprints` |
| `VoiceTranscribeClient.enrollVoiceprint` | `POST /api/voiceprints/enroll` |
| `VoiceTranscribeClient.renameVoiceprint` | `PUT /api/voiceprints/{id}/name` |
| `VoiceTranscribeClient.deleteVoiceprint` | `DELETE /api/voiceprints/{id}` |

Source files live in the [BetterAINote repo](https://github.com/MapleEve/BetterAINote)
under `src/lib/transcription/providers/voice-transcribe-provider.ts` and
`src/lib/voice-transcribe/client.ts`.
