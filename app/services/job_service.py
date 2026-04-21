"""Job management and background transcription worker.

Owns:
- _LRUJobsDict: bounded thread-safe LRU store for job states
- jobs: the singleton in-memory job registry
- _gpu_sem: semaphore that serialises GPU access to one transcription at a time
- run_transcription: the background worker function
"""

import json
import logging
import threading
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from config import (
    DENOISE_MODEL,
    DENOISE_SNR_THRESHOLD,
    JOBS_MAX_CACHE,
    TRANSCRIPTIONS_DIR,
    VOICEPRINT_THRESHOLD,
)
from services.audio_service import convert_to_wav, maybe_denoise, register_hash

logger = logging.getLogger(__name__)

# CQ-C1: counter used to periodically rebuild AS-norm cohort inside the
# transcription worker so it becomes active without requiring a server restart.
_cohort_rebuild_counter: dict = {}


# ---------------------------------------------------------------------------
# Status persistence helpers (AR-C2)
# ---------------------------------------------------------------------------


def _write_status(
    job_id: str,
    status: str,
    error: str | None = None,
    filename: str | None = None,
) -> None:
    """Write job status to disk for persistence across process restarts."""
    status_path = TRANSCRIPTIONS_DIR / job_id / "status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        payload = {
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "error": error,
        }
        if filename is not None:
            payload["filename"] = filename
        status_path.write_text(json.dumps(payload))
    except Exception as exc:
        logger.warning("Failed to write status.json for %s: %s", job_id, exc)


def recover_orphan_jobs() -> None:
    """Mark any in-progress jobs as failed if the process was restarted.

    Called once during application lifespan startup so that frontend polls
    receive a definitive terminal state instead of hanging on stale
    'transcribing'/'queued' statuses written by a previous process.
    """
    try:
        for status_path in TRANSCRIPTIONS_DIR.glob("*/status.json"):
            try:
                data = json.loads(status_path.read_text())
                if data.get("status") not in ("completed", "failed"):
                    data["status"] = "failed"
                    data["error"] = "Process restarted while job was in progress"
                    data["updated_at"] = datetime.now().isoformat()
                    status_path.write_text(json.dumps(data))
                    logger.info(
                        "AR-C2: marked orphan job %s as failed",
                        status_path.parent.name,
                    )
            except Exception as exc:
                logger.warning(
                    "AR-C2: could not recover orphan job at %s: %s", status_path, exc
                )
    except Exception as exc:
        logger.warning("AR-C2: orphan job recovery scan failed: %s", exc)


# ---------------------------------------------------------------------------
# Bounded LRU job store (CQ-H2 / PERF-C1)
# ---------------------------------------------------------------------------


class _LRUJobsDict:
    """Thread-safe LRU dict for job states with bounded size."""

    def __init__(self, maxsize: int = 200):
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def __setitem__(self, key, value):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
            self._d[key] = value
            if len(self._d) > self._maxsize:
                self._d.popitem(last=False)

    def __getitem__(self, key):
        with self._lock:
            return self._d[key]

    def __contains__(self, key):
        with self._lock:
            return key in self._d

    def get(self, key, default=None):
        with self._lock:
            return self._d.get(key, default)


# In-memory job status — bounded LRU (CQ-H2 / PERF-C1)
jobs: _LRUJobsDict = _LRUJobsDict(maxsize=JOBS_MAX_CACHE)

# Serialise GPU work: only one transcription runs at a time.
# Concurrent HTTP uploads are fine; they queue here before touching the GPU.
_gpu_sem = threading.Semaphore(1)


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------


def run_transcription(
    job_id: str,
    audio_path: Path,
    language: str,
    min_speakers: int,
    max_speakers: int,
    pipeline,
    voiceprint_db,
    denoise_model: str = None,
    snr_threshold: float = None,
    file_hash: str = None,
    no_repeat_ngram_size: int = 0,
):
    """Background transcription worker.

    Accepts *pipeline* and *voiceprint_db* as explicit arguments (injected by
    the route handler from app.state) to avoid global-state coupling and make
    the function testable in isolation.
    """
    # Track intermediate files so they can be cleaned up on both success and
    # failure. Initialise to audio_path so the cleanup guard (path != audio_path)
    # is safe even if an exception fires before the variables are reassigned.
    wav_path: Path = audio_path
    clean_path: Path = audio_path
    try:
        jobs[job_id]["status"] = "converting"
        _write_status(job_id, "converting", filename=audio_path.name)
        wav_path = convert_to_wav(audio_path)

        jobs[job_id]["status"] = "queued"
        _write_status(job_id, "queued")
        with _gpu_sem:
            _intermediate = (
                "denoising"
                if (denoise_model or DENOISE_MODEL) != "none"
                else "transcribing"
            )
            jobs[job_id]["status"] = _intermediate
            _write_status(job_id, _intermediate)
            clean_path = maybe_denoise(wav_path, denoise_model, snr_threshold)

            # DF peaks at ~15 GB reserved in PyTorch's CUDA cache.
            # ctranslate2 (Whisper) calls cudaMalloc directly and sees the OS
            # free memory — not PyTorch's allocator pool — so it OOMs unless we
            # explicitly flush the cache before Whisper cold-loads.
            try:
                import gc as _gc

                import torch as _torch

                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception as exc:
                logger.warning("pre-whisper CUDA cache flush failed: %s", exc)

            jobs[job_id]["status"] = "transcribing"
            _write_status(job_id, "transcribing")
            result = pipeline.process(
                str(clean_path),
                raw_audio_path=str(wav_path),
                language=language,
                min_speakers=min_speakers or None,
                max_speakers=max_speakers or None,
                no_repeat_ngram_size=no_repeat_ngram_size or None,
            )

        # Release cached CUDA memory so the next queued job has headroom
        try:
            import gc as _gc

            import torch as _torch

            _gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception as exc:
            logger.warning("post-pipeline CUDA cache flush failed: %s", exc)

        # Delete intermediate files — keep only the original uploaded file.
        # clean_path is the denoised WAV (may equal wav_path if denoising was skipped).
        # wav_path is the converted WAV (may equal audio_path if input was already WAV).
        if clean_path != wav_path:
            clean_path.unlink(missing_ok=True)
        if wav_path != audio_path:
            wav_path.unlink(missing_ok=True)

        # Match speakers against voiceprint DB
        jobs[job_id]["status"] = "identifying"
        _write_status(job_id, "identifying")
        speaker_map = {}
        for spk_label, embedding in result["speaker_embeddings"].items():
            spk_id, spk_name, sim = voiceprint_db.identify(
                embedding, threshold=VOICEPRINT_THRESHOLD
            )
            speaker_map[spk_label] = {
                "matched_id": spk_id,
                "matched_name": spk_name or spk_label,
                "similarity": round(sim, 4),
                "embedding_key": spk_label,
            }

        # [CQ-H6] 若所有 turn 均短于 MIN_EMBED_DURATION，embeddings 为空 → 不产生 speaker_map。
        # 记录明确 warning，让前端可以区分"无可登记 speaker"并避免传 'undefined' 字符串。
        warning = None
        if not speaker_map:
            warning = "no_speakers_detected"
            logger.warning(
                "Job %s produced no speaker embeddings (all turns < min duration)",
                job_id,
            )

        # Consolidate multiple diarization clusters that resolved to the same
        # enrolled speaker. Pick the cluster with the highest similarity as the
        # canonical label; remap all others to it so one person appears under a
        # single label rather than as separate SPEAKER_XX entries.
        _id_to_clusters: dict = {}
        for _lbl, _info in speaker_map.items():
            _mid = _info["matched_id"]
            if _mid is not None:
                _id_to_clusters.setdefault(_mid, []).append((_lbl, _info["similarity"]))

        _cluster_remap: dict[str, str] = {}
        for _mid, _cluster_list in _id_to_clusters.items():
            _cluster_list.sort(key=lambda x: x[1], reverse=True)
            _canonical_lbl = _cluster_list[0][0]
            for _lbl, _ in _cluster_list[1:]:
                _cluster_remap[_lbl] = _canonical_lbl
                logger.info(
                    "Job %s: merged cluster %s → %s (same enrolled speaker %s)",
                    job_id,
                    _lbl,
                    _canonical_lbl,
                    _mid,
                )

        # Build final segments with remapped speaker labels
        segments = []
        for i, seg in enumerate(result["segments"]):
            spk_label = seg["speaker"]
            canonical_label = _cluster_remap.get(spk_label, spk_label)
            match = speaker_map.get(canonical_label, speaker_map.get(spk_label, {}))
            out = {
                "id": i,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker_label": canonical_label,
                "speaker_id": match.get("matched_id"),
                "speaker_name": match.get("matched_name", canonical_label),
                "similarity": match.get("similarity", 0),
            }
            # Forward word-level timestamps when forced alignment produced them
            # (0.3.0+). Absent when the language has no alignment model or
            # alignment failed — clients must treat the key as optional.
            if seg.get("words"):
                out["words"] = seg["words"]
            segments.append(out)

        # Derive unique_speakers from resolved speaker names (ordered by first
        # appearance in the transcript, deduplicated). Enrolled speakers appear
        # under their enrolled name; unidentified clusters keep their raw label.
        _seen_spk: set = set()
        resolved_unique_speakers: list = []
        for seg in segments:
            name = seg["speaker_name"]
            if name not in _seen_spk:
                _seen_spk.add(name)
                resolved_unique_speakers.append(name)

        # Save transcription result
        effective_denoise = (denoise_model or DENOISE_MODEL).strip().lower()
        effective_snr = (
            snr_threshold if snr_threshold is not None else DENOISE_SNR_THRESHOLD
        )
        tr = {
            "id": job_id,
            "filename": audio_path.name,
            "created_at": datetime.now().isoformat(),
            "status": "completed",
            "language": language,
            "segments": segments,
            "speaker_map": speaker_map,
            "unique_speakers": resolved_unique_speakers,
            "params": {
                "language": language or "auto",
                "denoise_model": effective_denoise,
                "snr_threshold": effective_snr,
                "voiceprint_threshold": VOICEPRINT_THRESHOLD,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
                "no_repeat_ngram_size": no_repeat_ngram_size or 0,
            },
        }
        if warning is not None:
            tr["warning"] = warning

        tr_dir = TRANSCRIPTIONS_DIR / job_id
        tr_dir.mkdir(exist_ok=True)
        (tr_dir / "result.json").write_text(
            json.dumps(tr, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Save raw embeddings for later enrollment
        import numpy as np

        for spk_label, emb in result["speaker_embeddings"].items():
            np.save(tr_dir / f"emb_{spk_label}.npy", emb)

        if file_hash:
            register_hash(file_hash, job_id)

        # CQ-C1: After each successful transcription, check if AS-norm cohort
        # should be rebuilt. Every 10th job (or when cohort is absent) we rebuild
        # so that newly enrolled speakers contribute to normalization without
        # requiring a server restart.
        try:
            _cohort_rebuild_counter[0] = _cohort_rebuild_counter.get(0, 0) + 1
            if voiceprint_db.cohort_size == 0 or _cohort_rebuild_counter[0] % 10 == 0:
                voiceprint_db.build_cohort_from_transcriptions(str(TRANSCRIPTIONS_DIR))
                logger.info(
                    "AS-norm cohort rebuilt: size=%d", voiceprint_db.cohort_size
                )
        except Exception as exc:
            logger.warning("cohort rebuild failed: %s", exc)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = tr
        _write_status(job_id, "completed")
        logger.info(
            "Job %s completed: %d segments, %d speakers",
            job_id,
            len(segments),
            len(speaker_map),
        )

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        _write_status(job_id, "failed", error=str(e))
        # Best-effort cleanup of intermediate files on failure.
        try:
            if clean_path != wav_path:
                clean_path.unlink(missing_ok=True)
            if wav_path != audio_path:
                wav_path.unlink(missing_ok=True)
        except Exception:
            pass
