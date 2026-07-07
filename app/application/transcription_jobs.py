"""Application-level transcription job orchestration."""

import logging
import time
from pathlib import Path

from application.admission import release_transcription_admission
from config import (
    TRANSCRIPTIONS_DIR,
    VOICEPRINT_THRESHOLD,
)
from infra.audio import register_hash
from infra.job_persistence import write_job_status
from infra.job_runtime import (
    run_serialized_gpu_work,
    unregister_in_flight,
    update_runtime_job,
)

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
        update_runtime_job(job_id, {"status": status})
        extra_filename = audio_path.name if status == "converting" else None
        write_job_status(job_id, status, filename=extra_filename)

    job_started = time.perf_counter()
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

        update_runtime_job(job_id, {"status": "completed", "result": tr})
        write_job_status(job_id, "completed")
        logger.info(
            "transcription_job_timing status=completed elapsed_s=%.3f segment_count=%d speaker_count=%d",
            time.perf_counter() - job_started,
            len(tr.get("segments", [])),
            len(tr.get("speaker_map", {})),
        )
        if file_hash:
            unregister_in_flight(file_hash, job_id)

    except Exception as e:
        logger.exception(
            "transcription_job_timing status=failed elapsed_s=%.3f error_type=%s",
            time.perf_counter() - job_started,
            e.__class__.__name__,
        )
        update_runtime_job(job_id, {"status": "failed", "error": str(e)})
        write_job_status(job_id, "failed", error=str(e))
        if file_hash:
            unregister_in_flight(file_hash, job_id)
    finally:
        release_transcription_admission(job_id)
