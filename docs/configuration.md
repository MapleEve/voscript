# voscript Configuration Reference

voscript is a FastAPI service that transcribes meeting audio using faster-whisper (ASR), pyannote 3.1 (speaker diarization), WeSpeaker ResNet34 (voiceprint / speaker ID), DeepFilterNet3 (optional noise reduction), and OSD overlapped speech detection.

---

## Environment Variables

All variables are read at container startup. Changes require a container restart.

| Variable | Default | Description |
|----------|---------|-------------|
| `DATA_DIR` | `/data` | Root directory for uploads, transcriptions, and voiceprints. Subdirs `uploads/`, `transcriptions/`, and `voiceprints/` are created automatically. |
| `HF_TOKEN` | _(none)_ | HuggingFace access token. Required to download gated models: `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`, and `pyannote/wespeaker-voxceleb-resnet34-LM`. A missing or unauthorized token raises HTTP 403 on the first call to any of these models. |
| `WHISPER_MODEL` | `large-v3` | faster-whisper model size. Also accepts `medium`, `small`, `base`, `tiny` and their `.en` variants. If `/models/faster-whisper-<size>` exists on disk the local copy is used; otherwise the model is downloaded from HuggingFace. |
| `DEVICE` | `cuda` | Compute device passed to faster-whisper, pyannote, and WeSpeaker. Use `cpu` for CPU-only hosts (compute type switches to `int8` automatically). |
| `VOICEPRINT_THRESHOLD` | `0.75` | Base cosine-similarity threshold for speaker identification. The actual per-speaker threshold is adaptive — see [Feature: Adaptive Voiceprint Threshold](#feature-adaptive-voiceprint-threshold). |
| `DENOISE_MODEL` | `none` | Noise reduction backend. Accepted values: `none`, `deepfilternet`, `noisereduce`. When set to anything other than `none`, the SNR gate is applied before invoking the model. |
| `DENOISE_SNR_THRESHOLD` | `10.0` | SNR gate threshold (dB). Audio at or above this level is considered clean and the denoise step is skipped. See [Feature: Noise Reduction](#feature-noise-reduction-snr-gate). |
| `API_KEY` | _(none)_ | Bearer token for API authentication. When unset the service accepts unauthenticated requests (a warning is logged). When set, all `/api/*` routes require `Authorization: Bearer <key>` or `X-API-Key: <key>`. The web UI (`/`), `/healthz`, `/docs`, `/redoc`, `/openapi.json`, and `/static/*` remain public. |
| `MAX_UPLOAD_BYTES` | `2147483648` (2 GB) | Maximum size of a single audio upload in bytes. Requests exceeding this limit are rejected with HTTP 413 and the partial file is deleted. |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated list of allowed CORS origins (e.g. `https://app.example.com,https://admin.example.com`). Defaults to wildcard. |


---

## API Request Parameters (POST /api/transcribe)

The endpoint accepts `multipart/form-data`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | _(required)_ | Audio file to transcribe. Any format supported by ffmpeg is accepted (mp3, m4a, opus, wav, flac, ogg, mp4, etc.). The file is converted to 16 kHz mono WAV before processing. |
| `language` | string | `zh` | BCP-47 language code for transcription. Pass an empty string or omit to enable Whisper auto-detection. Common values: `zh` (Mandarin), `en` (English), `ja` (Japanese). |
| `min_speakers` | integer | `0` | Minimum number of speakers to pass to pyannote diarization. `0` means no constraint (pyannote infers). |
| `max_speakers` | integer | `0` | Maximum number of speakers to pass to pyannote diarization. `0` means no constraint. Setting this when the participant count is known reduces diarization errors. |
| `denoise_model` | string | `none` | Per-request noise reduction override. Accepted values: `none`, `deepfilternet`, `noisereduce`. Overrides the `DENOISE_MODEL` environment variable for this request only. The SNR gate still applies. |
| `snr_threshold` | float | _(global)_ | Per-request SNR gate override (dB). Overrides `DENOISE_SNR_THRESHOLD` for this request only. |
| `osd` | boolean | `false` | Enable overlapped speech detection. When `true`, each output segment receives a `has_overlap` boolean field indicating whether two or more speakers were detected speaking simultaneously at that point. Adds one extra model pass using `pyannote/segmentation-3.0`. |

### Response structure

The endpoint returns immediately with `{"id": "<job_id>", "status": "queued"}`. Poll `GET /api/jobs/<job_id>` for progress and results.

Job statuses in order: `queued` → `converting` → `denoising` (if applicable) → `transcribing` → `identifying` → `completed` (or `failed`).

When `status == "completed"` the response includes a `result` object:

```json
{
  "id": "tr_20260419_120000_abc123",
  "filename": "meeting.m4a",
  "created_at": "2026-04-19T12:00:00",
  "status": "completed",
  "language": "zh",
  "segments": [
    {
      "id": 0,
      "start": 1.234,
      "end": 4.567,
      "text": "今天的议题是...",
      "speaker_label": "SPEAKER_00",
      "speaker_id": "spk_a1b2c3d4",
      "speaker_name": "Maple",
      "similarity": 0.7135,
      "has_overlap": false,
      "words": [
        {"word": "今天", "start": 1.234, "end": 1.560, "score": 0.9812}
      ]
    }
  ],
  "speaker_map": {
    "SPEAKER_00": {
      "matched_id": "spk_a1b2c3d4",
      "matched_name": "Maple",
      "similarity": 0.7135,
      "embedding_key": "SPEAKER_00"
    }
  },
  "unique_speakers": ["SPEAKER_00", "SPEAKER_01"],
  "params": {
    "language": "zh",
    "denoise_model": "none",
    "snr_threshold": 10.0,
    "voiceprint_threshold": 0.75,
    "osd": false,
    "min_speakers": 0,
    "max_speakers": 0
  }
}
```

Notes:
- `words` is present only when WhisperX forced alignment succeeded for the detected language.
- `has_overlap` is present only when `osd=true` was passed in the request.
- `speaker_id` and `speaker_name` are `null` / the raw `speaker_label` when no voiceprint match was found above threshold.
- `params` records the effective settings used for this specific job. `denoise_model` and `snr_threshold` reflect the per-request override if one was supplied, otherwise the container defaults. This field makes each result self-contained — no separate config endpoint is needed.

---

## Feature: Adaptive Voiceprint Threshold

### What it does

`VOICEPRINT_THRESHOLD` sets only the **base** threshold. The actual effective threshold applied per-candidate is computed adaptively based on how consistent that speaker's enrolled embeddings are across sessions:

- **1 enrolled sample** (freshly enrolled): threshold is relaxed by a fixed 0.05. A single-sample enrollment has no cross-session variance data, so the fixed relaxation prevents the "one sample never matches again" failure mode.
- **2+ enrolled samples**: threshold is relaxed by `min(3.0 * intra_cluster_std, 0.10)`. Speakers whose embeddings vary more across sessions get a more lenient threshold automatically.
- **Absolute floor**: the effective threshold never drops below **0.60**, regardless of relaxation, to prevent false positives from degenerate clusters.

Formula: `effective = max(0.60, base - relaxation)`

With `VOICEPRINT_THRESHOLD=0.75` and 2 enrolled samples (spread = 0.0261), the effective threshold is **0.6717**.

As more enrollment samples are added, the intra-cluster spread stabilizes and the effective threshold converges automatically to the optimal value.

### A/B bench result (10 PLAUD Pin recordings, 2026-04-19)

| Recording | Similarity | Static 0.75 | Adaptive 0.6717 | Change |
|-----------|-----------|-------------|-----------------|--------|
| plaud_1 | 0.7160 | miss | hit | new identification |
| plaud_2 | 0.9099 | hit | hit | unchanged |
| plaud_3 | 0.7135 | miss | hit | new identification |
| plaud_4 | 0.0000 | miss | miss | unchanged (absent) |
| plaud_5 | 0.9078 | hit | hit | unchanged |
| plaud_6 | 0.0000 | miss | miss | unchanged (absent) |
| plaud_7 | 0.0000 | miss | miss | unchanged (absent) |
| plaud_8 | 0.7626 | hit | hit | unchanged |
| plaud_9 | 0.7737 | hit | hit | unchanged |
| plaud_10 | 0.9126 | hit | hit | unchanged |

| Threshold mode | Recall (n=10) | False positives |
|----------------|---------------|-----------------|
| Static 0.75 | 5/10 (50%) | 0 |
| Adaptive (effective 0.6717) | 7/10 (70%) | 0 |

The two newly identified recordings (plaud_1, plaud_3) had similarities of 0.716 and 0.714 — genuine participation in low-noise meeting rooms, falling just below the static threshold. The three recordings with similarity near 0.00 (speaker absent or silent) were correctly rejected by both modes.

---

## Feature: Noise Reduction (SNR Gate)

### What it does

When `DENOISE_MODEL` is set to anything other than `none`, voscript estimates the signal-to-noise ratio of each uploaded recording before deciding whether to apply the noise reduction model. The estimation uses a lightweight energy-based heuristic (30 ms frames, bottom 20% of frames treated as the noise floor) that adds negligible latency.

If the estimated SNR is at or above `DENOISE_SNR_THRESHOLD` (default **10.0 dB**, configurable via env or per-request `snr_threshold`), the audio is considered clean and the denoise step is skipped entirely. The log line reads:

```
DeepFilterNet skipped (SNR=19.7dB, clean audio)
```

This gate exists because DeepFilterNet was designed for environments with SNR below 20 dB. Applying it to already-clean audio introduces spectral distortion that degrades both ASR transcript quality and speaker embedding fidelity.

### SNR measurements — PLAUD Pin test batch (2026-04-19)

| Recording | SNR (dB) | Gate at 15 dB | Gate at 10 dB |
|-----------|----------|---------------|---------------|
| plaud_1 | 18.8 | applied | skipped |
| plaud_2 | 20.7 | skipped | skipped |
| plaud_3 | 26.0 | skipped | skipped |
| plaud_4 | 21.1 | skipped | skipped |
| plaud_5 | 20.1 | skipped | skipped |
| plaud_6 | 22.4 | skipped | skipped |
| plaud_7 | 20.1 | skipped | skipped |
| plaud_8 | 19.7 | applied | skipped |
| plaud_9 | 25.7 | skipped | skipped |
| plaud_10 | 12.4 | applied | skipped |

All 10 PLAUD Pin recordings have SNR >= 10 dB. With a threshold of 10 dB every one of them would be skipped, meaning `DENOISE_MODEL=deepfilternet` has effectively no impact on PLAUD Pin recordings when using that threshold.

### Why applying DF to PLAUD Pin recordings is harmful

When DeepFilterNet was applied to PLAUD Pin recordings (before the SNR gate was tuned), two problems were observed:

**Segment fragmentation**: Whisper's VAD splits audio into far more fragments after DF processing.

| Recording | Segments (no DF) | Segments (with DF) | Increase |
|-----------|------------------|--------------------|----------|
| plaud_8 (13.5 min) | 206 | 505 | +145% |
| plaud_10 (17.5 min) | 155 | 324 | +109% |
| plaud_9 (DF skipped at SNR 25.7 dB) | 351 | 352 | +0.3% |

Fragmented segments are more frequently misattributed by pyannote. plaud_9, where DF was correctly skipped, shows near-zero fragmentation, confirming the cause.

After the SNR threshold was lowered to 10 dB, plaud_8 was re-run and fragmentation dropped from +145% to +1.5%. plaud_10 (SNR 12.4 dB) is also now skipped: the log confirms `DeepFilterNet skipped (SNR=12.4dB, clean audio)`.

**Word-level content divergence**: Total character count difference is near zero (-0.1%, neutral), but specific words recognized diverge substantially (proxy CER 20–91% across recordings). DF alters the spectral fingerprint of clean speech, causing Whisper to produce different vocabulary even though overall output length is similar.

For comparison, plaud_9 (DF skipped) showed a proxy CER of only 4.3%, consistent with Whisper's natural non-determinism baseline.

---

## Feature: Overlapped Speech Detection (OSD)

### What it does

When `osd=true` is passed in a transcription request, voscript runs pyannote `OverlappedSpeechDetection` (backed by `pyannote/segmentation-3.0`) over the audio. The model detects intervals where two or more speakers are simultaneously active. Each output segment is annotated with `"has_overlap": true/false` based on whether the segment midpoint falls within a detected overlap interval.

The segmentation model is shared with the diarization pipeline and requires no additional download beyond the initial `HF_TOKEN` acceptance. OSD adds one extra model pass over the audio file.

### A/B bench result (10 PLAUD Pin recordings, 2026-04-19)

| Recording | Speakers | Segments | Overlap segs | Seg overlap% | Duration overlap% |
|-----------|----------|----------|--------------|-------------|-------------------|
| plaud_1 | 2 | 117 | 7 | 6.0% | 3.9% |
| plaud_2 | 3 | 162 | 19 | 11.7% | 8.9% |
| plaud_3 | 2 | 109 | 1 | 0.9% | 0.4% |
| plaud_4 | 2 | 232 | 33 | 14.2% | 8.7% |
| plaud_5 | 2 | 238 | 27 | 11.3% | 9.4% |
| plaud_6 | 2 | 212 | 18 | 8.5% | 4.8% |
| plaud_7 | 2 | 151 | 13 | 8.6% | 5.9% |
| plaud_8 | 3 | 222 | 34 | 15.3% | 10.7% |
| plaud_9 | 3 | 362 | 24 | 6.6% | 6.1% |
| plaud_10 | 2 | 352 | 34 | 9.7% | 4.3% |
| **Total** | | **2157** | **210** | **9.7%** | |

The 9.7% segment overlap rate is within the normal range for natural meeting conversation (academic baseline 10–15%). Three-speaker recordings average 11.2% overlap; two-speaker conversations average 8.7%.

The `has_overlap` field is produced reliably. No further speaker-separation pipeline (SepFormer, MossFormer2-SS) is warranted at the current overlap rate. If future recording scenarios shift toward multi-party roundtables (projected overlap rate >20%), a separation pipeline should be re-evaluated.

---

## Recommended Configuration (PLAUD Pin)

PLAUD Pin recordings have high SNR (12–26 dB) and clean built-in noise reduction. DeepFilterNet provides no benefit and introduces fragmentation. The recommended configuration keeps denoising disabled.

```yaml
# docker-compose.yml (relevant env section)
services:
  voscript:
    environment:
      - HF_TOKEN=hf_your_token_here
      - WHISPER_MODEL=large-v3
      - DEVICE=cuda
      - VOICEPRINT_THRESHOLD=0.75       # adaptive relaxation applied automatically
      - DENOISE_MODEL=none              # PLAUD Pin audio is already clean
      - API_KEY=your_secret_key_here
      - DATA_DIR=/data
      - MAX_UPLOAD_BYTES=2147483648
```

Example API call:

```bash
curl -X POST https://your-host/api/transcribe \
  -H "Authorization: Bearer your_secret_key_here" \
  -F "file=@meeting.m4a" \
  -F "language=zh" \
  -F "min_speakers=0" \
  -F "max_speakers=0" \
  -F "denoise_model=none" \
  -F "osd=false"
```

---

## Recommended Configuration (Noisy Environments)

For recordings made with phone microphones, far-field microphones, in cafes, or other high-noise environments (expected SNR typically below 10 dB), enable DeepFilterNet. The SNR gate engages it only when the recording actually needs it.

```yaml
# docker-compose.yml (relevant env section)
services:
  voscript:
    environment:
      - HF_TOKEN=hf_your_token_here
      - WHISPER_MODEL=large-v3
      - DEVICE=cuda
      - VOICEPRINT_THRESHOLD=0.75
      - DENOISE_MODEL=deepfilternet     # default for this deployment
      - API_KEY=your_secret_key_here
      - DATA_DIR=/data
```

To enable denoising for a single request without changing the container default:

```bash
curl -X POST https://your-host/api/transcribe \
  -H "Authorization: Bearer your_secret_key_here" \
  -F "file=@cafe_recording.m4a" \
  -F "language=zh" \
  -F "denoise_model=deepfilternet" \
  -F "osd=true"
```

The SNR gate (default 10 dB) means:
- Recordings with SNR >= 10 dB: DF is skipped automatically, no fragmentation risk.
- Recordings with SNR < 10 dB (typical cafe/street/phone audio): DF is applied.

Setting `DENOISE_MODEL=deepfilternet` at the container level is safe for mixed environments — the gate protects clean recordings automatically.

For the `noisereduce` backend: lightweight stationary noise reduction with no GPU requirement. The SNR gate still applies.
