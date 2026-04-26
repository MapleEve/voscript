# Full Configuration and Tuning Reference

[简体中文](./configuration.zh.md) | **English**

This is the public configuration index for VoScript v0.7.4. It covers the
environment variables that the current code reads, the per-request override
semantics of `POST /api/transcribe`, and internal defaults that are documented
for operators but are not stable public knobs yet. Do not assume a Whisper,
diarization, or AS-norm env var exists unless it is listed here.

## Configuration Sources and Precedence

| Layer | Example | Precedence |
| --- | --- | --- |
| API request field | `denoise_model=deepfilternet`, `snr_threshold=8` | Per-job only, wins over service env |
| Container environment | `.env` injected through `docker-compose.yml` | Service-level default |
| Code default | `app/config.py` | Fallback when env is empty or invalid |

`POST /api/transcribe` currently exposes only `language`, `min_speakers`,
`max_speakers`, `denoise_model`, `snr_threshold`, and `no_repeat_ngram_size`.
Other pipeline settings may have internal defaults, but they are not public API
parameters yet.

## Service Basics

| Variable | Default | Effect |
| --- | --- | --- |
| `API_KEY` | empty | When set, all endpoints except `/`, `/healthz`, `/docs`, `/redoc`, `/openapi.json`, and `/static/*` require `Authorization: Bearer <key>` or `X-API-Key: <key>`. |
| `ALLOW_NO_AUTH` | `0` | Used only when `API_KEY` is empty. `1` acknowledges unauthenticated mode and suppresses the startup warning; it does not add protection. |
| `CORS_ALLOW_ORIGINS` | `*` | Comma-separated CORS origins. Narrow this before exposing the service outside a trusted network. |
| `HOST_PORT` | `8780` | Host port published by compose; not an app runtime env var. |
| `MAX_UPLOAD_BYTES` | `2147483648` | Per-upload byte cap. Larger uploads return `413` and the partial file is removed. |
| `DATA_DIR` | `/data` | In-container data root for transcriptions, uploads, and voiceprints. Compose mounts host `./data` to `/data` by default. |
| `MODEL_CACHE_DIR` | `./models` | Host model-cache directory used by compose, mounted to `/cache` and read-only `/models`. |
| `APP_UID` / `APP_GID` | `1000` / `1000` | Container runtime user. Host `DATA_DIR` and `MODEL_CACHE_DIR` must be writable by this uid/gid. |
| `DEVICE` | `cuda` | Pipeline inference device. Use `cpu` for CPU-only, macOS, or non-NVIDIA hosts. |
| `CUDA_VISIBLE_DEVICES` | `0` | NVIDIA GPU selection. Use an empty value together with `DEVICE=cpu` for CPU-only mode. |
| `FFMPEG_TIMEOUT_SEC` | `1800` | ffmpeg conversion timeout in seconds; timeout returns `504`. |
| `JOBS_MAX_CACHE` | `200` | In-memory job LRU limit. Evicted completed jobs remain queryable from disk `status.json` / `result.json`. |

`MODELS_DIR` and `LANGUAGE` are defined in the config module, but v0.7.4's main
HTTP transcription path does not use them as stable public tuning knobs:
Whisper local checkpoint lookup still expects `/models/faster-whisper-<WHISPER_MODEL>`,
and default language should be controlled with the request `language` field or
left empty for auto-detection.

## Hugging Face and Model Cache

| Variable | Default | Effect |
| --- | --- | --- |
| `HF_TOKEN` | empty | Token for pyannote / WeSpeaker gated models. Accept the relevant Hugging Face model terms first. |
| `HF_ENDPOINT` | `https://huggingface.co` | Hugging Face Hub endpoint; use a trusted mirror in restricted networks. |
| `HF_HUB_DISABLE_XET` | `1` | Bypasses hf-xet/CAS downloads by default. Set `0` only when your environment supports hf-xet reliably. |
| `HF_HUB_ETAG_TIMEOUT` | `3` | Hub metadata timeout in seconds, so slow networks fall back to local cache quickly. |
| `HF_HOME` / `HUGGINGFACE_HUB_CACHE` / `TORCH_HOME` / `XDG_CACHE_HOME` | `/cache`-based paths | Dockerfile cache defaults. Usually configure the host mount through `MODEL_CACHE_DIR` instead of overriding these one by one. |

faster-whisper first looks for `/models/faster-whisper-<WHISPER_MODEL>`; if it
does not exist, it loads by model name. pyannote and WeSpeaker first try a
complete local Hugging Face snapshot and fall back to Hub loading only when the
cache is incomplete.

## Whisper / ASR

| Setting | Default | Supported Today |
| --- | --- | --- |
| `WHISPER_MODEL` | `large-v3` | Service env. Supports `tiny`, `base`, `small`, `medium`, `large-v3`, and other faster-whisper model names. |
| `DEVICE` | `cuda` | Service env. `cuda` uses `float16`; `cpu` uses `int8`. Compute type is not separately configurable yet. |
| API `language` | auto-detect | Per-request field. Empty means auto-detect and use the Mandarin-oriented initial prompt. |
| API `no_repeat_ngram_size` | `0` | Per-request field. Values `>=3` are passed to faster-whisper to suppress n-gram repetition; non-integers return `422`. |

Current internal ASR defaults are `beam_size=5`, `vad_filter=True`,
`vad_parameters.min_silence_duration_ms=500`, and `condition_on_previous_text=False`.
These do not have env or API fields in v0.7.4. Do not configure nonexistent
variables such as `WHISPER_BEAM_SIZE`, `WHISPER_COMPUTE_TYPE`, or `WHISPER_VAD_*`.

## Denoising

| Setting | Default | Effect |
| --- | --- | --- |
| `DENOISE_MODEL` | `none` | Service default backend: `none`, `deepfilternet`, or `noisereduce`. Unknown values log a warning and skip denoising. |
| `DENOISE_SNR_THRESHOLD` | `10.0` | DeepFilterNet SNR gate in dB. When `deepfilternet` is selected, audio estimated at or above this value is skipped to avoid degrading clean recordings; `noisereduce` does not use this gate. |
| API `denoise_model` | omitted | Omitted means inherit `DENOISE_MODEL`; explicit `none` disables denoising for this job only. |
| API `snr_threshold` | omitted | Omitted means inherit `DENOISE_SNR_THRESHOLD`; explicit values override the DeepFilterNet SNR gate for this job only. |

v0.7.4 defaults to `DENOISE_MODEL=none` for clean meeting-recorder audio. Enable
`deepfilternet` or `noisereduce` only for noisy environments, either per job or
as a service default. If you need clean recordings to be skipped automatically,
use `deepfilternet`; `noisereduce` runs whenever it is selected.

## Diarization and Alignment

| Setting | Default | Effect |
| --- | --- | --- |
| API `min_speakers` / `max_speakers` | `0` | Per-request speaker-count bounds. `0` means auto and is not passed to pyannote. |
| `PYANNOTE_MIN_DURATION_OFF` | `0.5` | pyannote `_binarize.min_duration_off`, used to merge short pauses and reduce over-segmentation. If the pyannote object does not support it, the service logs a warning and continues. |
| `WHISPERX_ALIGN_DISABLED_LANGUAGES` | empty | Comma-separated languages that skip forced alignment when no model override is present. Use only as a temporary operational fallback. |
| `WHISPERX_ALIGN_MODEL_MAP` | empty | Comma-separated `lang=model` overrides, for example `zh=org/model`. |
| `WHISPERX_ALIGN_MODEL_DIR` | empty | Optional alignment model directory; passed through only when the installed WhisperX supports that parameter. |
| `WHISPERX_ALIGN_CACHE_ONLY` | `0` | When `1`, requests cache-only alignment model loading, only when supported by the installed WhisperX. |

Alignment is optional metadata. On success, results may include
`alignment.status=succeeded` and `segments[].words`. If disabled or failed, the
job still completes; `words` may be absent and `alignment` records `skipped` or
`failed` using sanitized metadata. Clients must treat both fields as optional.

## Embedding

| Variable | Default | Effect |
| --- | --- | --- |
| `EMBEDDING_DIM` | `256` | Voiceprint vector dimension used for DB and AS-norm cohort shape checks. Do not mix existing stores across dimensions. |
| `MIN_EMBED_DURATION` | `1.5` | Diarization turns shorter than this are ignored for speaker embedding extraction. |
| `MAX_EMBED_DURATION` | `10.0` | Longer turns are clipped to this window before embedding extraction. |

Each speaker cluster uses up to the 10 longest usable chunks to produce an
averaged embedding. Very short, fragmented, or noisy turns reduce enrollment and
matching quality.

## Voiceprints and AS-norm

| Item | Default | Notes |
| --- | --- | --- |
| `VOICEPRINT_THRESHOLD` | `0.75` | Base threshold for raw cosine mode. The effective threshold adapts by sample count and `sample_spread`. |
| Raw single-sample relaxation | `0.05` | One-sample speakers default to an effective threshold around `0.70`. Internal default, not env. |
| Raw spread relaxation | `3.0 * sample_spread`, capped at `0.10` | Multi-sample speakers with larger sample spread get a moderate relaxation. Internal default. |
| Raw absolute floor | `0.60` | Raw cosine auto-naming never accepts below this value. Internal default. |
| AS-norm activation | `10` cohort embeddings | When cohort size is below 10, `ASNormScorer.score()` falls back to raw cosine. Internal default. |
| AS-norm base | `0.5` | Z-score-like operating point once the cohort is large enough; not raw cosine. Internal default. |
| AS-norm top-1/top-2 margin | `0.05` | If the best normalized candidate is too close to the second candidate, the speaker remains unnamed. Internal default. |
| AS-norm cohort `top_n` | `200` | Number of nearest cohort impostors used for AS-norm statistics, capped by cohort size. Internal default. |

`similarity` depends on cohort state:

- Cohort < 10 or AS-norm unavailable: `similarity` is raw cosine, usually in `[-1, 1]`.
- Cohort >= 10: `similarity` is an AS-norm normalized score and may exceed `1`
  or be negative.
- Only `speaker_id != null` means the candidate passed the effective threshold
  for the current mode; do not display `similarity` as a percentage.

Cohort lifecycle:

- On startup, an existing `data/transcriptions/asnorm_cohort.npy` is loaded directly.
- Otherwise, the service scans persisted transcription results and `emb_*.npy`
  files to build and save a cohort.
- After each enroll / update, the background `cohort-rebuild` thread wakes every
  60 seconds and rebuilds after the latest enrollment is at least 30 seconds old.
- v0.7.4 protects larger loaded or persisted cohorts during automatic rebuilds:
  clearing transcription results, having only a few embeddings, or having fewer
  source embeddings than the current cohort will not shrink the cohort automatically.
- `POST /api/voiceprints/rebuild-cohort` is an explicit manual rebuild and uses
  the currently available embeddings immediately.

## Result Contract

Stable anchors in completed transcription results:

- `status`: persisted result status is `completed`; the job endpoint can also
  report `queued`, `converting`, `denoising`, `transcribing`, `identifying`, or `failed`.
- `segments[].speaker_label`: raw pyannote cluster label, the stable key for
  enrollment and later correction.
- `segments[].speaker_name`: display name; falls back to `speaker_label` when
  unmatched, and is disambiguated when multiple clusters hit the same enrolled name.
- `segments[].speaker_id`: matched voiceprint ID, or `null`.
- `segments[].similarity`: speaker-level match score; raw cosine or AS-norm
  z-score depending on cohort state.
- `segments[].words`: optional word-level alignment.
- Top-level `alignment`: optional forced-alignment metadata, sanitized.
- Top-level `params`: effective per-job processing settings, including request
  overrides and service defaults used for this result.
- `speaker_map`: diarization cluster to voiceprint match map; manual segment
  corrections do not rewrite it.
- `unique_speakers`: deduplicated current segment display names.

New fields are added under the optional-field principle. Clients should ignore
unknown fields and tolerate missing `words`, `alignment`, and `warning`.

## v0.7.4 Validation Wording

v0.7.4 has internal live validation covering transcription cleanup while
retaining voiceprints: as long as the voiceprint DB and a loaded or persisted
AS-norm cohort remain, automatic background rebuilds do not shrink a larger
cohort to an empty or undersized one. New-voice enroll, cohort rebuild, probe
hit, and cleanup entrypoints were also covered. The current public validation
does not have trustworthy >=10 cohort evidence, so it only proves the voiceprint
API, cohort refresh entrypoint, and raw-cosine fallback are usable; it must not
claim the probe exercised the full AS-norm scoring path. Full AS-norm validation
requires cohort size >=10. Public documentation records only the behavioral
conclusion, not real task names, sample names, job IDs, speaker IDs, hosts, or
paths.

## Related Docs

- [Quickstart](./quickstart.en.md)
- [API reference](./api.en.md)
- [Voiceprint tuning reference](./voiceprint-tuning.en.md)
- [Changelog](./changelog.en.md)
