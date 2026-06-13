"""Application usecases for persisted transcription records and artifacts."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from infra.job_persistence import _atomic_write_json
from infra.job_status import normalize_status_payload

logger = logging.getLogger(__name__)

_TR_ID_RE = re.compile(r"^tr_[A-Za-z0-9_-]{1,64}$")
_SPK_ID_RE = re.compile(r"^spk_[A-Za-z0-9_-]{1,64}$")
_EXPORT_CTRL_RE = re.compile(r"[\r\n\x00-\x1f\x7f]+")


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


def _runtime_jobs():
    import infra.job_runtime as job_runtime

    return job_runtime.jobs


def _safe_tr_dir(tr_id: str, settings: TranscriptionRecordSettings) -> Path:
    if not _TR_ID_RE.match(tr_id):
        raise TranscriptionRecordError(
            "invalid_transcription_id",
            f"Invalid transcription ID format: {tr_id!r}",
        )

    root = settings.transcriptions_dir.resolve()
    path = (settings.transcriptions_dir / tr_id).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise TranscriptionRecordError(
            "invalid_transcription_id",
            "Path traversal detected",
        ) from exc
    return path


def get_job_status(
    job_id: str,
    *,
    settings: TranscriptionRecordSettings | None = None,
    runtime_jobs: Any | None = None,
) -> dict[str, Any]:
    settings = _settings_or_default(settings)
    runtime_jobs = _runtime_jobs() if runtime_jobs is None else runtime_jobs

    if job_id in runtime_jobs:
        job = runtime_jobs[job_id]
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

    tr_dir = _safe_tr_dir(job_id, settings)
    status_path = tr_dir / "status.json"
    result_path = tr_dir / "result.json"

    if not status_path.exists():
        raise TranscriptionRecordError("job_not_found", "Job not found")

    try:
        status_data = normalize_status_payload(json.loads(status_path.read_text()))
    except Exception as exc:
        logger.warning("Corrupt status.json for %s: %s", job_id, exc)
        raise TranscriptionRecordError("job_not_found", "Job not found") from exc

    current_status = status_data.get("status")

    if current_status == "completed" and result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            result = None
        return {
            "id": job_id,
            "status": "completed",
            "filename": status_data.get("filename"),
            "result": result,
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
    results: list[dict[str, Any]] = []
    for tr_dir in sorted(settings.transcriptions_dir.iterdir(), reverse=True):
        if not tr_dir.is_dir():
            continue
        result_file = tr_dir / "result.json"
        if not result_file.exists():
            continue
        try:
            data = json.loads(result_file.read_text(encoding="utf-8"))
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
            logger.warning(
                "Skipping corrupt result.json in %s: %s",
                tr_dir.name,
                exc,
            )
    return results


def load_transcription_result(
    tr_id: str,
    *,
    settings: TranscriptionRecordSettings | None = None,
) -> dict[str, Any]:
    settings = _settings_or_default(settings)
    result_file = _safe_tr_dir(tr_id, settings) / "result.json"
    if not result_file.exists():
        raise TranscriptionRecordError(
            "transcription_not_found",
            "Transcription not found",
        )
    try:
        return json.loads(result_file.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Corrupt result.json for %s: %s", tr_id, exc)
        raise TranscriptionRecordError(
            "corrupt_result",
            "Corrupt transcription artifact",
        ) from exc


def get_audio_artifact(
    tr_id: str,
    *,
    settings: TranscriptionRecordSettings | None = None,
) -> AudioArtifact:
    settings = _settings_or_default(settings)
    data = load_transcription_result(tr_id, settings=settings)
    filename = _safe_audio_filename(data.get("filename"))
    audio_file = _safe_upload_path(filename, settings)
    if not audio_file.exists():
        raise TranscriptionRecordError(
            "missing_audio",
            "Original audio file not found",
        )
    return AudioArtifact(path=audio_file, filename=filename)


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

    result_file = _safe_tr_dir(tr_id, settings) / "result.json"
    data = load_transcription_result(tr_id, settings=settings)

    segment = next((s for s in data["segments"] if s["id"] == seg_id), None)
    if segment is None:
        raise TranscriptionRecordError("segment_not_found", "Segment not found")

    segment["speaker_name"] = speaker_name
    segment["speaker_id"] = speaker_id or None
    data["unique_speakers"] = sorted(
        set(s["speaker_name"] for s in data["segments"] if s.get("speaker_name"))
    )

    _atomic_write_json(result_file, data, ensure_ascii=False, indent=2)
    return {"ok": True}


def build_export_payload(
    tr_id: str,
    export_format: str = "srt",
    *,
    settings: TranscriptionRecordSettings | None = None,
) -> ExportPayload:
    settings = _settings_or_default(settings)
    result_file = _safe_tr_dir(tr_id, settings) / "result.json"
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


def _safe_audio_filename(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise TranscriptionRecordError(
            "corrupt_result",
            "Corrupt transcription artifact",
        )

    posix_path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    if (
        value in {".", ".."}
        or posix_path.is_absolute()
        or windows_path.is_absolute()
        or posix_path.name != value
        or windows_path.name != value
    ):
        raise TranscriptionRecordError(
            "corrupt_result",
            "Corrupt transcription artifact",
        )
    return value


def _safe_upload_path(filename: str, settings: TranscriptionRecordSettings) -> Path:
    root = settings.uploads_dir.resolve()
    audio_file = (settings.uploads_dir / filename).resolve()
    try:
        audio_file.relative_to(root)
    except ValueError as exc:
        raise TranscriptionRecordError(
            "corrupt_result",
            "Corrupt transcription artifact",
        ) from exc
    return audio_file
