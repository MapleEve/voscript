"""Application-level orchestration for transcription upload submission."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from application.admission import (
    AdmissionBudget,
    AdmissionRejectedError,
    admit_transcription_in_flight,
    find_in_flight_transcription,
    release_transcription_admission,
    reserve_transcription_admission,
)
from application.transcription_jobs import run_transcription
from infra.audio import safe_log_filename

logger = logging.getLogger(__name__)
_MISSING = object()
Thread = threading.Thread


class _RuntimeJobsProxy:
    """Resolve the active runtime job store at use time.

    Test clients in this repo intentionally reload app modules under different
    DATA_DIR values. A proxy avoids pinning this usecase to a stale infra module.
    """

    @staticmethod
    def _store():
        import infra.job_runtime as job_runtime

        return job_runtime.jobs

    def __setitem__(self, key, value):
        self._store()[key] = value

    def __getitem__(self, key):
        return self._store()[key]

    def __contains__(self, key):
        return key in self._store()

    def get(self, key, default=None):
        return self._store().get(key, default)

    def pop(self, key, default=_MISSING):
        if default is _MISSING:
            return self._store().pop(key)
        return self._store().pop(key, default)

    def values_snapshot(self) -> tuple:
        return self._store().values_snapshot()

    def __len__(self):
        return len(self._store())


jobs = _RuntimeJobsProxy()


class UploadStream(Protocol):
    filename: str | None

    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True)
class TranscriptionSubmissionCommand:
    file: UploadStream
    pipeline: Any
    voiceprint_db: Any
    language: str | None = None
    min_speakers: int = 0
    max_speakers: int = 0
    denoise_model: str | None = None
    snr_threshold: float | None = None
    no_repeat_ngram_size: int = 0


@dataclass(frozen=True)
class TranscriptionSubmissionSettings:
    max_upload_bytes: int
    upload_chunk: int
    max_active_jobs: int
    max_in_flight_jobs: int
    uploads_dir: Path
    transcriptions_dir: Path


@dataclass(frozen=True)
class TranscriptionSubmissionResult:
    job_id: str
    status: str
    deduplicated: bool = False


class TranscriptionSubmissionError(RuntimeError):
    """Typed application error for submission failures."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _new_job_id() -> str:
    return f"tr_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def default_submission_settings() -> TranscriptionSubmissionSettings:
    import config

    return TranscriptionSubmissionSettings(
        max_upload_bytes=config.MAX_UPLOAD_BYTES,
        upload_chunk=config.UPLOAD_CHUNK,
        max_active_jobs=config.TRANSCRIPTION_MAX_ACTIVE_JOBS,
        max_in_flight_jobs=config.TRANSCRIPTION_MAX_IN_FLIGHT_JOBS,
        uploads_dir=config.UPLOADS_DIR,
        transcriptions_dir=config.TRANSCRIPTIONS_DIR,
    )


def _admission_budget(settings: TranscriptionSubmissionSettings) -> AdmissionBudget:
    return AdmissionBudget(
        max_active_jobs=settings.max_active_jobs,
        max_in_flight_jobs=settings.max_in_flight_jobs,
    )


def _submission_error_from_admission(
    exc: AdmissionRejectedError,
) -> TranscriptionSubmissionError:
    return TranscriptionSubmissionError(exc.reason, str(exc))


def _safe_upload_name(file: UploadStream) -> str:
    name = PurePosixPath(file.filename or "upload").name or "upload"
    return safe_log_filename(name) or "upload"


async def _default_upload_saver(
    file: UploadStream,
    save_path: Path,
    max_bytes: int,
    chunk_size: int,
) -> tuple[int, str]:
    from infra.audio import save_upload_and_hash

    return await save_upload_and_hash(file, save_path, max_bytes, chunk_size)


def _default_hash_lookup(file_hash: str) -> str | None:
    from infra.audio import lookup_hash

    return lookup_hash(file_hash)


def _default_status_writer(*args, **kwargs) -> bool:
    from infra.job_persistence import _write_status

    return _write_status(*args, **kwargs)


def _unregister_in_flight(file_hash: str, job_id: str) -> bool:
    import infra.job_runtime as job_runtime

    return job_runtime.unregister_in_flight(file_hash, job_id)


def _discard_bootstrap_job(
    job_id: str,
    save_path: Path,
    *,
    transcriptions_dir: Path,
) -> None:
    jobs.pop(job_id, _MISSING)
    save_path.unlink(missing_ok=True)
    tr_dir = transcriptions_dir / job_id
    (tr_dir / "status.json").unlink(missing_ok=True)
    try:
        tr_dir.rmdir()
    except OSError:
        pass


async def submit_transcription_upload(
    command: TranscriptionSubmissionCommand,
    *,
    settings: TranscriptionSubmissionSettings | None = None,
    job_id_factory: Callable[[], str] = _new_job_id,
    thread_factory: Callable[..., Any] | None = None,
    worker: Callable[..., Any] | None = None,
    status_writer: Callable[..., bool] | None = None,
    upload_saver: Callable[[UploadStream, Path, int, int], Awaitable[tuple[int, str]]]
    | None = None,
    hash_lookup: Callable[[str], str | None] | None = None,
) -> TranscriptionSubmissionResult:
    """Accept an upload and bootstrap a durable background transcription job."""

    settings = settings or default_submission_settings()
    thread_factory = thread_factory or Thread
    worker = worker or run_transcription
    status_writer = status_writer or _default_status_writer
    upload_saver = upload_saver or _default_upload_saver
    hash_lookup = hash_lookup or _default_hash_lookup
    language = command.language.strip() if command.language else None
    job_id = job_id_factory()
    safe_filename = _safe_upload_name(command.file)
    save_path = settings.uploads_dir / f"{job_id}_{safe_filename}"

    try:
        _size, file_hash = await upload_saver(
            command.file,
            save_path,
            settings.max_upload_bytes,
            settings.upload_chunk,
        )
    except ValueError as exc:
        save_path.unlink(missing_ok=True)
        raise TranscriptionSubmissionError("upload_too_large", str(exc)) from exc

    existing_id = hash_lookup(file_hash)
    if existing_id:
        save_path.unlink(missing_ok=True)
        logger.info(
            "Dedup hit: %s already transcribed as %s", safe_filename, existing_id
        )
        return TranscriptionSubmissionResult(
            job_id=existing_id,
            status="completed",
            deduplicated=True,
        )

    existing_job = find_in_flight_transcription(file_hash) if file_hash else None
    if existing_job:
        save_path.unlink(missing_ok=True)
        logger.info(
            "In-flight dedup: %s already processing as %s",
            safe_filename,
            existing_job,
        )
        return TranscriptionSubmissionResult(
            job_id=existing_job,
            status="queued",
            deduplicated=True,
        )

    active_reserved = False
    budget = _admission_budget(settings)
    try:
        reserve_transcription_admission(job_id, budget)
        active_reserved = True
    except AdmissionRejectedError as exc:
        save_path.unlink(missing_ok=True)
        raise _submission_error_from_admission(exc) from exc

    jobs[job_id] = {
        "status": "queued",
        "filename": safe_filename,
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    if not status_writer(job_id, "queued", filename=safe_filename):
        _discard_bootstrap_job(
            job_id,
            save_path,
            transcriptions_dir=settings.transcriptions_dir,
        )
        if active_reserved:
            release_transcription_admission(job_id)
        raise TranscriptionSubmissionError(
            "job_state_persist_failed",
            "Failed to persist job state — disk error, retry later",
        )

    if file_hash:
        try:
            registration = admit_transcription_in_flight(file_hash, job_id, budget)
        except AdmissionRejectedError as exc:
            _discard_bootstrap_job(
                job_id,
                save_path,
                transcriptions_dir=settings.transcriptions_dir,
            )
            if active_reserved:
                release_transcription_admission(job_id)
            raise _submission_error_from_admission(exc) from exc
        if registration.existing_job_id:
            _discard_bootstrap_job(
                job_id,
                save_path,
                transcriptions_dir=settings.transcriptions_dir,
            )
            if active_reserved:
                release_transcription_admission(job_id)
            logger.info(
                "In-flight dedup: %s already processing as %s",
                safe_filename,
                registration.existing_job_id,
            )
            return TranscriptionSubmissionResult(
                job_id=registration.existing_job_id,
                status="queued",
                deduplicated=True,
            )
        if not registration.registered:
            _discard_bootstrap_job(
                job_id,
                save_path,
                transcriptions_dir=settings.transcriptions_dir,
            )
            if active_reserved:
                release_transcription_admission(job_id)
            raise TranscriptionSubmissionError(
                "in_flight_registration_failed",
                "Failed to register in-flight transcription",
            )

    thread = thread_factory(
        target=worker,
        args=(
            job_id,
            save_path,
            language,
            command.min_speakers,
            command.max_speakers,
            command.pipeline,
            command.voiceprint_db,
            command.denoise_model,
            command.snr_threshold,
            file_hash,
            command.no_repeat_ngram_size if command.no_repeat_ngram_size >= 3 else 0,
        ),
        daemon=True,
    )
    try:
        thread.start()
    except Exception as exc:
        logger.exception("Failed to start transcription thread for %s", job_id)
        # Durable bootstrap has already succeeded. Preserve the failed job and
        # status.json as the observable old-router-compatible record, while
        # releasing transient upload, in-flight, and admission state below.
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = "Failed to start background transcription"
        status_writer(job_id, "failed", error=str(exc), filename=safe_filename)
        save_path.unlink(missing_ok=True)
        if file_hash:
            _unregister_in_flight(file_hash, job_id)
        if active_reserved:
            release_transcription_admission(job_id)
        raise TranscriptionSubmissionError(
            "thread_start_failed",
            "Failed to start background transcription — retry later",
        ) from exc

    return TranscriptionSubmissionResult(job_id=job_id, status="queued")
