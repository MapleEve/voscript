"""Application usecases for persisted transcription records and artifacts."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from infra.job_runtime import get_runtime_job
from infra.transcription_records import (
    FilesystemTranscriptionRecordRepository,
    TranscriptionRecordStorageError,
)

logger = logging.getLogger(__name__)

_SPK_ID_RE = re.compile(r"^spk_[A-Za-z0-9_-]{1,64}$")
_EXPORT_CTRL_RE = re.compile(r"[\r\n\x00-\x1f\x7f]+")
_MISSING = object()


@dataclass(frozen=True)
class TranscriptionRecordSettings:
    transcriptions_dir: Path
    uploads_dir: Path


@dataclass(frozen=True)
class AudioArtifact:
    path: Path
    filename: str


@dataclass(frozen=True)
class ExportPayload:
    media_type: str
    filename: str
    text: str | None = None
    file_path: Path | None = None


class TranscriptionRecordError(RuntimeError):
    """Typed application error for record read/write failures."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def default_record_settings() -> TranscriptionRecordSettings:
    import config

    return TranscriptionRecordSettings(
        transcriptions_dir=config.TRANSCRIPTIONS_DIR,
        uploads_dir=config.UPLOADS_DIR,
    )


def _settings_or_default(
    settings: TranscriptionRecordSettings | None,
) -> TranscriptionRecordSettings:
    return settings or default_record_settings()


def _repository(
    settings: TranscriptionRecordSettings,
) -> FilesystemTranscriptionRecordRepository:
    return FilesystemTranscriptionRecordRepository(
        transcriptions_dir=settings.transcriptions_dir,
        uploads_dir=settings.uploads_dir,
    )


def _raise_record_error(exc: TranscriptionRecordStorageError) -> None:
    raise TranscriptionRecordError(exc.reason, str(exc)) from exc


def _lookup_runtime_job(job_id: str, runtime_jobs: Any | None) -> Any:
    if runtime_jobs is None:
        return get_runtime_job(job_id, _MISSING)
    if job_id in runtime_jobs:
        return runtime_jobs[job_id]
    return _MISSING


def get_job_status(
    job_id: str,
    *,
    settings: TranscriptionRecordSettings | None = None,
    runtime_jobs: Any | None = None,
) -> dict[str, Any]:
    settings = _settings_or_default(settings)
    repository = _repository(settings)

    job = _lookup_runtime_job(job_id, runtime_jobs)
    if job is not _MISSING:
        response = {
            "id": job_id,
            "status": job["status"],
            "filename": job.get("filename"),
        }
        if job["status"] == "completed":
            response["result"] = job["result"]
        elif job["status"] == "failed":
            response["error"] = job.get("error")
        return response

    try:
        snapshot = repository.job_status_snapshot(job_id)
    except TranscriptionRecordStorageError as exc:
        _raise_record_error(exc)
    if snapshot is None:
        raise TranscriptionRecordError("job_not_found", "Job not found")

    status_data = snapshot.status
    current_status = status_data.get("status")

    if current_status == "completed" and snapshot.result_exists:
        return {
            "id": job_id,
            "status": "completed",
            "filename": status_data.get("filename"),
            "result": snapshot.result,
        }

    if current_status not in ("completed", "failed"):
        return {
            "id": job_id,
            "status": "failed",
            "error": "Process restarted while job was in progress",
            "filename": status_data.get("filename"),
        }

    return {
        "id": job_id,
        "status": current_status,
        "error": status_data.get("error"),
        "filename": status_data.get("filename"),
    }


def list_transcriptions(
    *,
    settings: TranscriptionRecordSettings | None = None,
) -> list[dict[str, Any]]:
    settings = _settings_or_default(settings)
    repository = _repository(settings)
    results: list[dict[str, Any]] = []
    for data in repository.iter_transcription_results():
        try:
            results.append(
                {
                    "id": data["id"],
                    "filename": data["filename"],
                    "created_at": data["created_at"],
                    "segment_count": len(data["segments"]),
                    "speaker_count": len(data.get("unique_speakers", [])),
                }
            )
        except Exception as exc:
            logger.warning("Skipping malformed transcription result: %s", exc)
    return results


def load_transcription_result(
    tr_id: str,
    *,
    settings: TranscriptionRecordSettings | None = None,
) -> dict[str, Any]:
    settings = _settings_or_default(settings)
    repository = _repository(settings)
    try:
        return repository.load_result(tr_id)
    except TranscriptionRecordStorageError as exc:
        _raise_record_error(exc)


def get_audio_artifact(
    tr_id: str,
    *,
    settings: TranscriptionRecordSettings | None = None,
) -> AudioArtifact:
    settings = _settings_or_default(settings)
    repository = _repository(settings)
    data = load_transcription_result(tr_id, settings=settings)
    try:
        audio = repository.uploaded_audio_artifact(data.get("filename"))
    except TranscriptionRecordStorageError as exc:
        _raise_record_error(exc)
    return AudioArtifact(path=audio.path, filename=audio.filename)


def reassign_speaker(
    tr_id: str,
    seg_id: int,
    speaker_name: str,
    speaker_id: str | None = None,
    *,
    voiceprint_db: Any | None = None,
    settings: TranscriptionRecordSettings | None = None,
) -> dict[str, bool]:
    settings = _settings_or_default(settings)
    repository = _repository(settings)
    if speaker_id:
        if not _SPK_ID_RE.match(speaker_id):
            raise TranscriptionRecordError(
                "invalid_speaker_id",
                "Invalid speaker_id format",
            )
        if voiceprint_db is None or voiceprint_db.get_speaker(speaker_id) is None:
            raise TranscriptionRecordError(
                "missing_voiceprint",
                f"Voiceprint {speaker_id} not found",
            )

    data = load_transcription_result(tr_id, settings=settings)

    segment = next((s for s in data["segments"] if s["id"] == seg_id), None)
    if segment is None:
        raise TranscriptionRecordError("segment_not_found", "Segment not found")

    segment["speaker_name"] = speaker_name
    segment["speaker_id"] = speaker_id or None
    data["unique_speakers"] = sorted(
        set(s["speaker_name"] for s in data["segments"] if s.get("speaker_name"))
    )

    try:
        repository.save_result(tr_id, data)
    except TranscriptionRecordStorageError as exc:
        _raise_record_error(exc)
    return {"ok": True}


def build_export_payload(
    tr_id: str,
    export_format: str = "srt",
    *,
    settings: TranscriptionRecordSettings | None = None,
) -> ExportPayload:
    settings = _settings_or_default(settings)
    repository = _repository(settings)
    data = load_transcription_result(tr_id, settings=settings)
    segments = data["segments"]

    if export_format == "srt":
        lines = []
        for index, segment in enumerate(segments, 1):
            start = _format_srt_time(segment["start"])
            end = _format_srt_time(segment["end"])
            speaker_name = _sanitize_export_speaker_name(segment.get("speaker_name"))
            lines.append(
                f"{index}\n{start} --> {end}\n[{speaker_name}] {segment['text']}\n"
            )
        return ExportPayload(
            text="\n".join(lines),
            media_type="text/srt",
            filename=f"{tr_id}.srt",
        )

    if export_format == "txt":
        lines = []
        for segment in segments:
            timestamp = _format_timestamp(segment["start"])
            speaker_name = _sanitize_export_speaker_name(segment.get("speaker_name"))
            lines.append(f"[{timestamp}] {speaker_name}: {segment['text']}")
        return ExportPayload(
            text="\n".join(lines),
            media_type="text/plain",
            filename=f"{tr_id}.txt",
        )

    if export_format == "json":
        try:
            result_file = repository.result_file_path(tr_id)
        except TranscriptionRecordStorageError as exc:
            _raise_record_error(exc)
        return ExportPayload(
            file_path=result_file,
            media_type="application/json",
            filename=f"{tr_id}.json",
        )

    raise TranscriptionRecordError(
        "unsupported_export_format",
        "Unsupported format. Use: srt, txt, json",
    )


def _format_srt_time(seconds: float) -> str:
    if seconds is None or seconds != seconds:
        seconds = 0.0
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{milliseconds:03d}"


def _format_timestamp(seconds: float) -> str:
    if seconds is None or seconds != seconds:
        seconds = 0.0
    seconds = max(0.0, float(seconds))
    minutes = int(seconds // 60)
    whole_seconds = int(seconds % 60)
    return f"{minutes:02d}:{whole_seconds:02d}"


def _sanitize_export_speaker_name(value: object) -> str:
    return _EXPORT_CTRL_RE.sub(" ", str(value or "")).strip()
