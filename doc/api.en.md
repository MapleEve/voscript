# API Reference

[简体中文](./api.zh.md) | **English**

All endpoints live under `http://<host>:8780`. JSON for data, `multipart/form-data`
for file uploads.

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
queued → converting → denoising (if DENOISE_MODEL ≠ none) → transcribing → identifying → completed
                                                                                              ↘ failed
```

The OpenPlaud(Maple) worker polls `/api/jobs/{id}` every 5 seconds and stops
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
| `denoise_model` | string | Optional. Noise reduction backend: `none` (default), `deepfilternet`, `noisereduce`. Overrides the `DENOISE_MODEL` container env for this request only. |
| `snr_threshold` | float | Optional. SNR gate threshold (dB) for this request only. Audio at or above this level skips denoising. Overrides `DENOISE_SNR_THRESHOLD`. |

Response (200):

```json
{ "id": "tr_20260418_080205_ea79b7", "status": "queued" }
```

If the uploaded file is byte-for-byte identical to a file from a previously completed job
(same SHA256), the endpoint returns the existing result immediately without re-running Whisper.
The response includes an extra field `deduplicated: true`:

```json
{ "id": "tr_existing_id", "status": "completed", "deduplicated": true }
```

The returned `id` works exactly like any other — poll `/api/jobs/{id}` or export as usual.

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

Example:

```bash
curl -X POST http://localhost:8780/api/transcribe \
     -H "Authorization: Bearer $API_KEY" \
     -F "file=@meeting.wav" \
     -F "language=en" \
     -F "max_speakers=4"
```

### `GET /api/jobs/{id}` — poll a job

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
    "params": {
      "language": "en",  // shows "auto" when no language was specified at submit time
      "denoise_model": "none",
      "snr_threshold": 10.0,
      "voiceprint_threshold": 0.75,
      "min_speakers": 0,
      "max_speakers": 0
    }
  }
}
```

**`speaker_label` is the raw pyannote label** — it never changes even when
an existing voiceprint was matched. Use it as the key for any later
enrollment or rename call.

`speaker_id` / `speaker_name`: when `similarity ≥ 0.75` the service has
auto-matched a registered voiceprint. Otherwise `speaker_id` is `null` and
`speaker_name` falls back to the raw label (e.g. `SPEAKER_00`).

**`words[]` is a new optional field added in 0.3.0** (WhisperX forced
alignment output). Each entry carries its own `start`/`end`/`score`.
Alignment for some Chinese utterances can fail; when it does, the key is
simply absent from the segment, the job still finishes. Clients that
don't recognize the field should just ignore it.

**`params`** records the effective settings used for this specific job,
including any per-request overrides. Makes each result self-contained —
no need to cross-reference the original request.

### `GET /api/transcriptions` — list past jobs

```json
[
  { "id": "tr_...", "filename": "...", "created_at": "...",
    "segment_count": 42, "speaker_count": 3 }
]
```

### `GET /api/transcriptions/{tr_id}` — full result

Same shape as the `result` field inside `GET /api/jobs/{id}`.

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
  { "id": "spk_61f24bd0", "name": "Alice",
    "sample_count": 3,
    "created_at": "2026-04-18T08:06:41.951819",
    "updated_at": "2026-04-18T09:17:02.113207" }
]
```

#### `POST /api/voiceprints/enroll`

Form fields:

| Field | Required | Description |
| --- | --- | --- |
| `tr_id` | ✅ | Transcription id, matches `result.id` |
| `speaker_label` | ✅ | **Must** be the raw `SPEAKER_XX` label, not the display name |
| `speaker_name` | ✅ | Display name, e.g. "Alice" |
| `speaker_id` | ❌ | Pass to update an existing voiceprint; omit to create |

Response:

```json
{ "action": "created | updated", "speaker_id": "spk_..." }
```

Example:

```bash
curl -X POST http://localhost:8780/api/voiceprints/enroll \
     -H "Authorization: Bearer $API_KEY" \
     -F "tr_id=tr_20260418_080205_ea79b7" \
     -F "speaker_label=SPEAKER_00" \
     -F "speaker_name=Alice"
```

#### `POST /api/voiceprints/rebuild-cohort`

Rebuilds the AS-norm impostor cohort matrix from all existing transcriptions. The service runs this automatically on startup; trigger it manually after bulk-ingesting new recordings.

Response:

```json
{ "cohort_size": 313, "saved_to": "/data/transcriptions/asnorm_cohort.npy" }
```

Since 0.5.0, the service auto-builds the AS-norm scoring matrix on startup from existing transcriptions. When active, voiceprint identification uses a normalized score (relative to the impostor distribution); the effective threshold is fixed at `0.5` and `VOICEPRINT_THRESHOLD` is ignored. Use `/api/voiceprints/rebuild-cohort` to refresh manually.

#### `PUT /api/voiceprints/{id}/name`

Form `name=<new name>`. Renames only; the embedding is unchanged.

#### `DELETE /api/voiceprints/{id}`

Removes the voiceprint permanently. Future recordings of that person will
not auto-match.

### `PUT /api/transcriptions/{tr_id}/segments/{seg_id}/speaker`

Manually reassign a single segment to a different speaker.

Form fields: `speaker_name` (required), `speaker_id` (optional).

## Error responses

| Code | Meaning |
| --- | --- |
| 400 | Missing or invalid request field |
| 401 | Missing or wrong API key |
| 404 | Unknown tr_id / speaker_id / missing embedding |
| 413 | Upload exceeded `MAX_UPLOAD_BYTES` (default 2 GiB) — see `/api/transcribe` |
| 500 | Server-side exception (check `docker logs voscript`) |

Body shape:

```json
{ "detail": "..." }
```

## OpenPlaud(Maple) mapping

| OpenPlaud(Maple) code | Endpoint called |
| --- | --- |
| `submitVoiceTranscribeJob` | `POST /api/transcribe` |
| `pollVoiceTranscribeJob` | `GET /api/jobs/{id}` |
| `VoiceTranscribeClient.listVoiceprints` | `GET /api/voiceprints` |
| `VoiceTranscribeClient.enrollVoiceprint` | `POST /api/voiceprints/enroll` |
| `VoiceTranscribeClient.renameVoiceprint` | `PUT /api/voiceprints/{id}/name` |
| `VoiceTranscribeClient.deleteVoiceprint` | `DELETE /api/voiceprints/{id}` |

Source files live in the [OpenPlaud(Maple) repo](https://github.com/MapleEve/openplaud)
under `src/lib/transcription/providers/voice-transcribe-provider.ts` and
`src/lib/voice-transcribe/client.ts`.
