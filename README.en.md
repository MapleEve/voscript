# openplaud-voice-transcribe

[简体中文](./README.md) | **English**

Self-hosted GPU transcription service with persistent speaker voiceprints.
Designed as the private backend for [OpenPlaud](https://github.com/MapleEve/openplaud),
but usable as a stand-alone FastAPI service.

```
Audio  ──►  faster-whisper large-v3  (transcription)
        ──►  pyannote 3.1             (speaker diarization)
        ──►  ECAPA-TDNN               (speaker embeddings)
        ──►  VoiceprintDB             (cosine match vs. enrolled speakers)
        ──►  timestamped text with identified speaker names
```

## Why a separate repo

OpenPlaud is a single-user control panel. The heavy work — loading
whisper/pyannote, keeping models resident in GPU memory, running diarization,
maintaining a voiceprint database — stays behind a private HTTP API so that
the public panel never ships a GPU model or a raw embedding to the browser.

This repo is that private API. OpenPlaud submits audio to it, polls for the
job, stores the transcript locally, and calls the voiceprint endpoints when
the user enrolls a speaker.

## Features

- Async job pipeline (`queued → converting → transcribing → identifying → completed`)
- Chinese + multilingual transcription (faster-whisper large-v3)
- Speaker diarization (pyannote 3.1)
- Persistent voiceprints: enroll once, auto-match in future recordings
  (cosine similarity ≥ 0.75)
- Stable HTTP contract consumed by OpenPlaud's
  [`voice-transcribe-provider.ts`](https://github.com/MapleEve/openplaud/blob/main/src/lib/transcription/providers/voice-transcribe-provider.ts)
  and [`voice-transcribe/client.ts`](https://github.com/MapleEve/openplaud/blob/main/src/lib/voice-transcribe/client.ts)
- Optional Bearer / `X-API-Key` auth on every `/api/*` route
- Minimal built-in web UI at `/` for manual testing

## Requirements

- Linux host with an NVIDIA GPU (~9 GB VRAM for whisper large-v3 + pyannote +
  ECAPA-TDNN). Tested on RTX 3090.
- Docker 24+ with the NVIDIA Container Toolkit.
- A HuggingFace access token. You must accept the terms for
  [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  and [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0),
  then create a token at <https://huggingface.co/settings/tokens>.

## Quick Start

```bash
git clone https://github.com/MapleEve/openplaud-voice-transcribe.git
cd openplaud-voice-transcribe

cp .env.example .env
# edit .env — at minimum set HF_TOKEN and API_KEY

docker compose --env-file .env up -d --build
curl -sf http://localhost:8780/healthz
```

First start downloads ~5 GB of model weights into `./models/` (or wherever
`MODEL_CACHE_DIR` points). They are cached across restarts.

## HTTP API

All endpoints return JSON. When `API_KEY` is set, every `/api/*` request
must include one of:

```
Authorization: Bearer <API_KEY>
X-API-Key: <API_KEY>
```

`GET /healthz`, `GET /`, and `/static/*` are always public.

### Submit a transcription

```
POST /api/transcribe
  multipart/form-data:
    file          audio file (wav/mp3/m4a/...)
    language      "zh" (default), "en", ...
    min_speakers  integer, 0 = auto
    max_speakers  integer, 0 = auto
→ 200 { "id": "tr_YYYYMMDD_HHMMSS_XXXXXX", "status": "queued" }
```

### Poll a job

```
GET /api/jobs/{id}
→ 200 {
    "id": "tr_...",
    "status": "queued" | "converting" | "transcribing" | "identifying"
            | "completed" | "failed",
    "filename": "...",
    "error": "...",                       // only when status = failed
    "result": {                            // only when status = completed
      "id": "tr_...",
      "language": "zh",
      "segments": [
        {
          "id": 0,
          "start": 0.0,          // seconds
          "end": 4.32,
          "text": "...",
          "speaker_label": "SPEAKER_00",   // raw pyannote label (stable for enroll)
          "speaker_id":    "spk_..." | null,
          "speaker_name":  "张三" | "SPEAKER_00",
          "similarity":    0.8421
        }
      ]
    }
  }
```

### Voiceprints

```
GET    /api/voiceprints
POST   /api/voiceprints/enroll       (tr_id, speaker_label, speaker_name, [speaker_id])
PUT    /api/voiceprints/{id}/name    (name)
DELETE /api/voiceprints/{id}
```

When enrolling, `speaker_label` MUST be the raw `SPEAKER_XX` label from the
job result, not the display `speaker_name` — the server keys cached embeddings
by the raw label that diarization produced.

### Export helpers

```
GET /api/transcriptions
GET /api/transcriptions/{tr_id}
GET /api/export/{tr_id}?format=srt|txt|json
PUT /api/transcriptions/{tr_id}/segments/{seg_id}/speaker  (speaker_name, [speaker_id])
```

## Wiring into OpenPlaud

In OpenPlaud settings, set:

- **Private transcription base URL**: `http://<host>:8780`
- **Private transcription API key**: the same `API_KEY` you set in `.env`

OpenPlaud then routes every recording through this service and stores the
transcript + speakers locally. See the OpenPlaud README for the full user
flow.

## Data layout on the host

```
data/
├── uploads/              # original uploads, keyed by job id
├── transcriptions/
│   └── tr_.../
│       ├── result.json   # full transcript + speaker map
│       └── emb_SPEAKER_XX.npy   # per-speaker embedding (used on enroll)
└── voiceprints/
    ├── index.json
    ├── spk_xxx_avg.npy
    └── spk_xxx_samples.npy
```

Back up `data/voiceprints/`. Everything else can be re-derived from the source
audio if you still have it.

## Security

Read [SECURITY.md](./SECURITY.md) before exposing this service. Short version:

1. Always set `API_KEY`. Don't expose `:8780` directly to the Internet.
2. `.env` is gitignored. Rotate `HF_TOKEN` if it ever lands in a log or image.
3. Voiceprints are biometric data. Treat `data/voiceprints/` accordingly.

## Development notes

- `app/main.py` is the FastAPI entrypoint. Auth middleware runs before every
  route; `/healthz`, `/`, `/static/*`, `/docs` are always public.
- `app/pipeline.py` wraps faster-whisper + pyannote + ECAPA-TDNN. It uses
  `use_auth_token` (the kwarg pyannote 3.1.1 expects) for HF downloads.
- `app/voiceprint_db.py` is a plain numpy-on-disk speaker database with
  running-average embeddings and cosine similarity matching.
- Version pins in `requirements.txt` are load-bearing:
  - `numpy<2` — pyannote 3.1.1 uses `np.NaN` which numpy 2.x removed
  - `huggingface_hub<0.24` — keeps the `use_auth_token` kwarg that pyannote
    3.1.1 calls

## License

MIT — see [LICENSE](./LICENSE).
