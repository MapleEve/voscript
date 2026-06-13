"""Filesystem repository for persisted transcription records."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

from infra.job_persistence import atomic_write_json
from infra.job_status import normalize_status_payload

logger = logging.getLogger(__name__)

_TR_ID_RE = re.compile(r"^tr_[A-Za-z0-9_-]{1,64}$")


class TranscriptionRecordStorageError(RuntimeError):
    """Infra storage error that application usecases map to typed errors."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class PersistedJobStatusSnapshot:
    status: dict[str, Any]
    result_exists: bool = False
    result: dict[str, Any] | None = None


@dataclass(frozen=True)
class UploadedAudioArtifact:
    path: Path
    filename: str


class FilesystemTranscriptionRecordRepository:
    """Read and write transcription record files under infra ownership."""

    def __init__(self, *, transcriptions_dir: Path, uploads_dir: Path) -> None:
        self.transcriptions_dir = transcriptions_dir
        self.uploads_dir = uploads_dir

    def job_status_snapshot(
        self,
        job_id: str,
    ) -> PersistedJobStatusSnapshot | None:
        tr_dir = self._safe_tr_dir(job_id)
        status_path = tr_dir / "status.json"
        result_path = tr_dir / "result.json"

        if not status_path.exists():
            return None

        try:
            status_data = normalize_status_payload(
                json.loads(status_path.read_text(encoding="utf-8"))
            )
        except Exception as exc:
            logger.warning("Corrupt status.json for %s: %s", job_id, exc)
            raise TranscriptionRecordStorageError(
                "job_not_found",
                "Job not found",
            ) from exc

        result_exists = result_path.exists()
        result = None
        if status_data.get("status") == "completed" and result_exists:
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                result = None

        return PersistedJobStatusSnapshot(
            status=status_data,
            result_exists=result_exists,
            result=result,
        )

    def iter_transcription_results(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for tr_dir in sorted(self.transcriptions_dir.iterdir(), reverse=True):
            if not tr_dir.is_dir():
                continue
            result_file = tr_dir / "result.json"
            if not result_file.exists():
                continue
            try:
                results.append(json.loads(result_file.read_text(encoding="utf-8")))
            except Exception as exc:
                logger.warning(
                    "Skipping corrupt result.json in %s: %s",
                    tr_dir.name,
                    exc,
                )
        return results

    def load_result(self, tr_id: str) -> dict[str, Any]:
        result_file = self.result_file_path(tr_id)
        if not result_file.exists():
            raise TranscriptionRecordStorageError(
                "transcription_not_found",
                "Transcription not found",
            )
        try:
            return json.loads(result_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Corrupt result.json for %s: %s", tr_id, exc)
            raise TranscriptionRecordStorageError(
                "corrupt_result",
                "Corrupt transcription artifact",
            ) from exc

    def save_result(self, tr_id: str, payload: dict[str, Any]) -> None:
        atomic_write_json(
            self.result_file_path(tr_id),
            payload,
            ensure_ascii=False,
            indent=2,
        )

    def result_file_path(self, tr_id: str) -> Path:
        return self._safe_tr_dir(tr_id) / "result.json"

    def uploaded_audio_artifact(self, filename_value: object) -> UploadedAudioArtifact:
        filename = self._safe_audio_filename(filename_value)
        audio_file = self._safe_upload_path(filename)
        if not audio_file.exists():
            raise TranscriptionRecordStorageError(
                "missing_audio",
                "Original audio file not found",
            )
        return UploadedAudioArtifact(path=audio_file, filename=filename)

    def _safe_tr_dir(self, tr_id: str) -> Path:
        if not _TR_ID_RE.match(tr_id):
            raise TranscriptionRecordStorageError(
                "invalid_transcription_id",
                f"Invalid transcription ID format: {tr_id!r}",
            )

        root = self.transcriptions_dir.resolve()
        path = (self.transcriptions_dir / tr_id).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise TranscriptionRecordStorageError(
                "invalid_transcription_id",
                "Path traversal detected",
            ) from exc
        return path

    def _safe_audio_filename(self, value: object) -> str:
        if not isinstance(value, str) or not value:
            raise TranscriptionRecordStorageError(
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
            raise TranscriptionRecordStorageError(
                "corrupt_result",
                "Corrupt transcription artifact",
            )
        return value

    def _safe_upload_path(self, filename: str) -> Path:
        root = self.uploads_dir.resolve()
        audio_file = (self.uploads_dir / filename).resolve()
        try:
            audio_file.relative_to(root)
        except ValueError as exc:
            raise TranscriptionRecordStorageError(
                "corrupt_result",
                "Corrupt transcription artifact",
            ) from exc
        return audio_file


__all__ = [
    "FilesystemTranscriptionRecordRepository",
    "PersistedJobStatusSnapshot",
    "TranscriptionRecordStorageError",
    "UploadedAudioArtifact",
]
