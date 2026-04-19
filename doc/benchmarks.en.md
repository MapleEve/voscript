# Benchmarks

[简体中文](./benchmarks.zh.md) | **English**

End-to-end wall-clock timing and resource usage on real audio. Every
measurement below is taken from an external HTTP client hitting
`POST /api/transcribe` + polling `/api/jobs/{id}`, plus container and
host-side resource sampling.

Raw 5-second samples are kept under [`benchmarks/`](./benchmarks/) as CSV.
Timestamps are **seconds relative to the client's submission**. Columns:
`t_offset_s, phase, cpu_pct, ram_gib, gpu_mem_mb, gpu_util_pct`.

---

## 1-hour Chinese meeting (voscript 0.3.0, warm)

**Input**: 57.6 minutes of Chinese meeting audio, mp3, 40 kbps, mono,
16 kHz, 23.5 MB.

**Environment**:
- Single 24 GB consumer-class NVIDIA GPU
- voscript 0.3.0 (WhisperX + faster-whisper large-v3 + pyannote 3.1 +
  WeSpeaker ResNet34 + sqlite-vec)
- Model weights already warm in the on-disk cache (**not** cold-start)
- `WHISPER_MODEL=large-v3`, `DEVICE=cuda`

Raw samples: [`benchmarks/1h-zh-meeting.csv`](./benchmarks/1h-zh-meeting.csv)
(184 rows).

### Wall-clock per phase

| Phase | Offset range | Duration |
| --- | --- | --- |
| Client multipart upload (23 MB) | `t = 0s → t = 1s` | **1 s** |
| `queued → transcribing` (container dispatch) | `t = 1s → t = 22s` | 21 s |
| **Transcribe** (faster-whisper large-v3 + VAD) | `t = 22s → t = 14m 7s` | **14 m 45 s** |
| **Align + Embed + Identify** (wav2vec2 + WeSpeaker + voiceprint DB) | `t = 14m 7s → t = 15m 42s` | **35 s** |
| **Total wall clock** | `t = 0s → t = 15m 42s` | **15 m 42 s** |

**Real-time factor RTF ≈ 3.7×** (57.6 min audio / 15.7 min wall time).

> A cold start (image just pulled, alignment model not downloaded) adds
> ~12–14 minutes on the first run to download the Chinese wav2vec2
> alignment model (~1 GB). Subsequent runs hit the cache and match the
> warm numbers above.

### Outputs

| Metric | Value |
| --- | --- |
| segments | **1 226** |
| segments with word-level timestamps (`words[]`) | **1 220 / 1 226** (99.5%) |
| total word-level timestamp entries | **13 149** |
| unique speakers identified | **8** (+ 21 `UNKNOWN` fallback segments, 1.7%) |
| coverage | 11.4 s → 3 458.7 s (≈ 57.5 min, near full coverage) |
| primary-speaker share | 721 / 1 226 ≈ **59%** (one dominant speaker, 7 others chiming in) |

### Resource usage (5 s sampling, 184 samples)

| Phase | n | CPU avg | CPU peak | RAM | GPU mem | GPU util avg | GPU util peak |
| --- | --- | --- | --- | --- | --- | --- | --- |
| idle (warm) | 10 | 72% | 287% | 1.92 GiB | 6.5 GiB | 19% | 98% |
| **transcribe** · 14 m 45 s | 130 | 121% | 586% | 2.08 GiB | 7.3 GiB | **21%** | 100% |
| **align + embed** · 35 s | 6 | 64% | 100% | 2.76 GiB | 8.1 GiB | **40%** | 63% |
| idle (post-completion) | 38 | 0% | 0% | 2.77 GiB | 8.5 GiB | 0.1% | 1% |

### What the numbers say

1. **Bottleneck is CPU + GIL, not GPU.** During transcribe GPU util
   averages 21% with brief 100% spikes; faster-whisper's VAD and
   tokenizer run on CPU, and the Python GIL adds another ceiling. Ways
   to push RTF down:
   - Raise `batch_size` (WhisperX default is 16)
   - Give faster-whisper more CPU workers
   - Enable flash attention
2. **Transcription dominates; alignment + voiceprints are basically
   free.** 1 hour of audio = 14 m 45 s of whisper; wav2vec2 +
   WeSpeaker + voiceprint identify together take 35 s.
3. **Plenty of VRAM headroom.** Peak 8.5 GiB out of 24 GiB — concurrent
   jobs on the same card are fine.
4. **Clean shutdown.** GPU util returns to 0 right after completion.
   Steady-state RAM 2.77 GiB, no drift, no residual inference.
5. **Word-alignment success rate 99.5%.** The 0.5% misses are mostly
   pure digits or ultra-short interjections — the Chinese wav2vec2
   model has no reliable character mapping for those tokens. On failure
   we drop the `words[]` field for that segment rather than crashing
   the job.

### One known threshold issue surfaced by this run

An enrolled voiceprint (1 sample) existed in the library before the
run. The engine **correctly** picked it as the best match for the
primary speaker (highest cosine across the 8 diarized clusters) — but
the maximum cosine was **0.7472**, 0.0028 short of the default 0.75
threshold, so the match was rejected and the segment fell back to its
raw `SPEAKER_XX` label.

Root cause:
- Only 1 sample at enrollment — the averaged embedding is noisy
- 0.75 is inherited from the ECAPA-TDNN era; WeSpeaker ResNet34's
  cosine distribution is slightly tighter, and 0.75 is a touch strict
  for a single-sample cross-session match

Mitigations (from 0.3.1):
- `VOICEPRINT_THRESHOLD` env var (default stays at `0.75`)
- Strategy: call `update_speaker` a few times in subsequent sessions so
  the averaged embedding stabilizes — two or more samples typically push
  cosine past 0.80 across sessions

**This issue was fully resolved in 0.4.0** — per-speaker adaptive threshold is now implemented. See [Section 2](#2-adaptive-voiceprint-threshold-ab-10-plaud-pin-recordings-2026-04-19).
- Medium-term: **per-speaker adaptive thresholds** (loosen based on the
  spread of that speaker's existing samples)

---

## 2. Adaptive Voiceprint Threshold A/B (10 PLAUD Pin recordings, 2026-04-19)

`VOICEPRINT_THRESHOLD` sets the base threshold (default 0.75). The engine
applies per-speaker adaptive relaxation based on intra-cluster standard
deviation:

- 1 enrolled sample → fixed -0.05 relaxation (single sample has no variance data)
- 2+ enrolled samples → `min(3 × intra_cluster_std, 0.10)` relaxation
- Absolute floor: effective threshold never drops below 0.60
- Formula: `effective = max(0.60, base - relaxation)`
- With 2 enrolled samples (spread = 0.0261): effective = max(0.60, 0.75 - 0.0783) = **0.6717**

**A/B table (10 recordings, 1 enrolled speaker)**:

| Recording | Similarity | Static 0.75 | Adaptive 0.6717 | Change |
|---|---|---|---|---|
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

**Summary**:

| Mode | Recall (n=10) | False positives |
|---|---|---|
| Static 0.75 | 5/10 (50%) | 0 |
| Adaptive (effective 0.6717) | 7/10 (70%) | 0 |

plaud_1 and plaud_3 (similarity 0.716 and 0.714) are genuine participation
segments recorded in low-noise meeting rooms, falling just below the static
threshold. The three recordings near 0.00 (speaker absent or silent) are
correctly rejected by both modes.

---

## 3. DeepFilterNet Harm on High-SNR Audio (2026-04-19)

**SNR measurements — 10 PLAUD Pin recordings**:

| Recording | SNR (dB) | Gate at 15 dB | Gate at 10 dB |
|---|---|---|---|
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

All 10 PLAUD Pin recordings have SNR ≥ 10 dB. With threshold = 10 dB, every
recording is skipped.

**Segment fragmentation when DF is applied to clean audio**:

| Recording | Segments (no DF) | Segments (with DF) | Increase |
|---|---|---|---|
| plaud_8 (13.5 min) | 206 | 505 | +145% |
| plaud_10 (17.5 min) | 155 | 324 | +109% |
| plaud_9 (DF skipped at SNR 25.7 dB) | 351 | 352 | +0.3% |

After lowering the SNR threshold to 10 dB and re-running plaud_8, fragmentation
dropped from +145% to +1.5%.

**Word-level content divergence**: Total character count difference is near
zero (-0.1%, neutral), but proxy CER ranges from 20–91% across recordings when
DF is applied. plaud_9 (DF skipped) showed only 4.3% proxy CER — consistent
with Whisper's natural non-determinism baseline.

Conclusion: `DENOISE_MODEL=none` is the correct setting for PLAUD Pin and
similar high-quality microphones. `DENOISE_SNR_THRESHOLD=10.0` ensures that
genuinely noisy recordings (SNR < 10 dB) still get processed when
`DENOISE_MODEL=deepfilternet`.

---

## 4. Overlapped Speech Detection (OSD) Statistics (2026-04-19)

**10 PLAUD Pin recordings with `osd=true`**:

| Recording | Speakers | Segments | Overlap segs | Seg overlap% | Duration overlap% |
|---|---|---|---|---|---|
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

A 9.7% segment overlap rate is within the normal range for natural meeting
conversation (academic baseline 10–15%). 3-speaker recordings average 11.2%
overlap; 2-speaker recordings average 8.7%. No speaker separation pipeline
is needed at current overlap rates.
