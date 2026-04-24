"""Application-level transcription job orchestration."""

import logging
from pathlib import Path

from config import (
    TRANSCRIPTIONS_DIR,
    VOICEPRINT_THRESHOLD,
)
from infra.audio import register_hash
from infra.job_persistence import _write_status
from infra.job_runtime import jobs, run_serialized_gpu_work, unregister_in_flight

logger = logging.getLogger(__name__)


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

    def _record_status(status: str) -> None:
        jobs[job_id]["status"] = status
        extra_filename = audio_path.name if status == "converting" else None
        _write_status(job_id, status, filename=extra_filename)

    try:

        def _process_pipeline():
            return pipeline.process(
                str(audio_path),
                language=language,
                min_speakers=min_speakers or None,
                max_speakers=max_speakers or None,
                no_repeat_ngram_size=no_repeat_ngram_size or None,
                voiceprint_db=voiceprint_db,
                voiceprint_threshold=VOICEPRINT_THRESHOLD,
                denoise_model=denoise_model,
                snr_threshold=snr_threshold,
                artifact_dir=TRANSCRIPTIONS_DIR / job_id,
                status_callback=_record_status,
            )

        result = run_serialized_gpu_work(_process_pipeline, logger=logger)

        tr = result.get("transcription")
        if tr is None:
            raise RuntimeError("Pipeline artifacts stage did not return transcription")

        if file_hash:
            register_hash(file_hash, job_id)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = tr
        _write_status(job_id, "completed")
        logger.info(
            "Job %s completed: %d segments, %d speakers",
            job_id,
            len(tr.get("segments", [])),
            len(tr.get("speaker_map", {})),
        )
        if file_hash:
            unregister_in_flight(file_hash, job_id)

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        _write_status(job_id, "failed", error=str(e))
        if file_hash:
            unregister_in_flight(file_hash, job_id)
