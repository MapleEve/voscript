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
- Medium-term: **per-speaker adaptive thresholds** (loosen based on the
  spread of that speaker's existing samples)
