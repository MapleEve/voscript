# Voiceprint Tuning Reference

This page lists the public tuning knobs that affect speaker matching and the
current internal defaults that are intentionally not API parameters yet.

## Environment Variables

| Name | Default | Scope | Notes |
| --- | ---: | --- | --- |
| `VOICEPRINT_THRESHOLD` | `0.75` | Raw cosine matching | Base threshold for raw cosine mode. The effective threshold is adjusted per enrolled speaker by sample count and sample spread. |
| `DATA_DIR` | `/data` | Storage | Parent directory for transcriptions, uploads, and `voiceprints/`. |
| `EMBEDDING_DIM` | `256` | Voiceprint DB | Embedding vector dimension used when creating/loading the vector index. Existing stores should not be mixed across dimensions. |
| `DENOISE_MODEL` | `none` | Transcription quality | Can change embeddings indirectly by changing the audio fed to diarization/embedding. |
| `DENOISE_SNR_THRESHOLD` | `10.0` | Transcription quality | Applies when denoising is enabled or requested. |

## API Parameters

| Endpoint | Parameter | Default | Notes |
| --- | --- | ---: | --- |
| `POST /api/transcribe` | `language` | auto | Affects ASR/alignment, not the voiceprint threshold directly. |
| `POST /api/transcribe` | `min_speakers`, `max_speakers` | `0` | Controls diarization bounds; bad bounds can create poor speaker embeddings. |
| `POST /api/transcribe` | `denoise_model`, `snr_threshold` | service defaults | Can change downstream embeddings and match quality. |
| `POST /api/transcribe` | `no_repeat_ngram_size` | `0` | ASR-only repetition guard, included here for complete transcription tuning. |
| `POST /api/voiceprints/enroll` | `speaker_name`, `speaker_label`, optional `speaker_id` | required/optional | Adds samples to the voiceprint library. More clean samples improve calibration. |
| `POST /api/voiceprints/rebuild-cohort` | none | n/a | Forces AS-norm cohort rebuild from persisted transcription embeddings. |

## Current Internal Defaults

These values are code defaults today. They are documented for operators, but
they are not stable public API knobs until explicitly exposed.

| Knob | Default | Effect |
| --- | ---: | --- |
| Raw single-sample relaxation | `0.05` | Raw cosine one-sample threshold is `base - 0.05` (`0.70` by default). |
| Raw spread relaxation | `3.0 * sample_spread`, capped at `0.10` | Raw cosine multi-sample speakers with larger spread get a lower threshold. |
| Raw absolute floor | `0.60` | Raw cosine matching never accepts below this threshold. |
| AS-norm cohort activation | `10` embeddings | Below this, scoring falls back to raw cosine and raw dynamic thresholding. |
| AS-norm operating threshold | `0.5` | Base z-score threshold used once cohort size is at least `10`. |
| AS-norm single-sample penalty | `+0.10` | A one-sample speaker needs at least `0.60` when the AS-norm base is `0.5`. |
| AS-norm unknown-spread penalty | `+0.05` | Legacy multi-sample rows without spread metadata are treated conservatively. |
| AS-norm low-sample penalty | `+0.025` per missing sample below `3` | Two-sample speakers need slightly stronger evidence than stable speakers. |
| AS-norm spread penalty | `0.50 * sample_spread`, capped at `0.10` | Noisy AS-norm enrollments need stronger evidence before auto-naming. |
| AS-norm stable relaxation | `-0.02` | Speakers with at least `3` samples and spread `<= 0.03` can match slightly below the base. |
| AS-norm top-1/top-2 margin | `0.05` | If the best AS-norm candidate is too close to the second candidate, the result stays unknown. |
| AS-norm cohort `top_n` | `200` | Number of nearest cohort impostors used for AS-norm statistics, capped by cohort size. |
| Cohort auto-rebuild loop | wake every `60s`, debounce `30s` | New enrollments normally enter AS-norm scoring within about `30-90s`. |

## AS-norm Tuning Guidance

- Keep AS-norm and raw cosine thresholds separate. AS-norm scores are z-scores,
  so raw cosine constants should not be copied directly.
- For production precision, avoid lowering the AS-norm single-sample threshold
  below `0.60` without an internal benchmark. A score around `0.5713` for a
  one-sample candidate is intentionally not enough to auto-name.
- Add clean samples before lowering thresholds. Three to five consistent
  samples are better than one aggressive threshold tweak.
- Increase the AS-norm margin when false accepts involve two similar enrolled
  speakers. Decrease it only if review volume is too high and benchmarked false
  accepts remain acceptable.
- Increase `top_n` only when the cohort is large and representative. Very small
  or biased cohorts should be fixed by rebuilding/expanding the cohort, not by
  increasing `top_n`.
- Treat `speaker_id = null` as "needs review", not as a failure. The service is
  deliberately conservative when evidence is sparse or ambiguous.
