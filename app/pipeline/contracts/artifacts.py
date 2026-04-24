"""Stable contracts for upload persistence and transcription artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class AsyncUploadReader(Protocol):
    """Minimal async file interface used by UploadFile and test doubles."""

    async def read(self, size: int = -1) -> bytes: ...


@dataclass(frozen=True, slots=True)
class UploadPersistenceRequest:
    """Describe how an uploaded audio file should be persisted and hashed."""

    file: AsyncUploadReader
    save_path: Path
    max_bytes: int
    chunk_size: int


@dataclass(frozen=True, slots=True)
class SavedUploadArtifact:
    """Result returned after persisting an upload and computing its hash."""

    path: Path
    size_bytes: int
    file_hash: str


@runtime_checkable
class AudioArtifactIndex(Protocol):
    """Stable slot for upload hashing and hash-index persistence."""

    async def persist_upload(
        self, request: UploadPersistenceRequest
    ) -> SavedUploadArtifact: ...

    def compute_file_hash(self, path: Path) -> str: ...

    def lookup(self, file_hash: str) -> str | None: ...

    def register(self, file_hash: str, artifact_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class TranscriptionArtifactWriteRequest:
    """Describe the result payload and embeddings to persist for a job."""

    output_dir: Path
    transcription: dict[str, Any]
    speaker_embeddings: dict[str, Any]


@dataclass(frozen=True, slots=True)
class PersistedTranscriptionArtifacts:
    """Paths written when a completed transcription is persisted."""

    result_path: Path
    embedding_paths: dict[str, Path]


@runtime_checkable
class TranscriptionArtifactStore(Protocol):
    """Stable slot for persisting completed transcription artifacts."""

    def persist_transcription(
        self, request: TranscriptionArtifactWriteRequest
    ) -> PersistedTranscriptionArtifacts: ...


__all__ = [
    "AsyncUploadReader",
    "AudioArtifactIndex",
    "PersistedTranscriptionArtifacts",
    "SavedUploadArtifact",
    "TranscriptionArtifactStore",
    "TranscriptionArtifactWriteRequest",
    "UploadPersistenceRequest",
]
