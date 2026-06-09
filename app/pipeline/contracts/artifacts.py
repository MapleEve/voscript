"""Stable contracts for upload persistence and transcription artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
import re
from typing import Any, Protocol, runtime_checkable

ARTIFACT_MANIFEST_VERSION = "artifact_manifest.v1"
ARTIFACT_MANIFEST_CATEGORIES = ("stable", "optional", "experimental")

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_UNSAFE_FILENAME_RE = re.compile(r"[\\/]|://")


def _public_safe_text(value: Any, field_name: str) -> str:
    text = _CONTROL_RE.sub(" ", str(value or "")).strip()
    if not text:
        raise ValueError(f"artifact manifest {field_name} must not be empty")
    return text


def _public_safe_filename(value: Any) -> str:
    text = _public_safe_text(value, "filename")
    # Filenames in the manifest are artifact names only, never host-local paths.
    if _UNSAFE_FILENAME_RE.search(text) or PurePosixPath(text).name != text:
        raise ValueError("artifact manifest filename must not include a path or URL")
    return text


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


@dataclass(frozen=True, slots=True)
class ArtifactManifestEntry:
    """Public-safe artifact descriptor embedded in completed results.

    This intentionally describes artifact names and roles without exposing
    host-local paths. Clients may ignore the whole manifest.
    """

    name: str
    filename: str
    role: str
    media_type: str
    required_for_result: bool = False
    speaker_label: str | None = None

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ArtifactManifestEntry":
        """Build an entry from a mapping while ignoring unknown fields."""

        return cls(
            name=_public_safe_text(payload.get("name"), "name"),
            filename=_public_safe_filename(payload.get("filename")),
            role=_public_safe_text(payload.get("role"), "role"),
            media_type=_public_safe_text(payload.get("media_type"), "media_type"),
            required_for_result=bool(payload.get("required_for_result", False)),
            speaker_label=(
                _public_safe_text(payload.get("speaker_label"), "speaker_label")
                if payload.get("speaker_label") is not None
                else None
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": _public_safe_text(self.name, "name"),
            "filename": _public_safe_filename(self.filename),
            "role": _public_safe_text(self.role, "role"),
            "media_type": _public_safe_text(self.media_type, "media_type"),
            "required_for_result": self.required_for_result,
        }
        if self.speaker_label is not None:
            payload["speaker_label"] = _public_safe_text(
                self.speaker_label, "speaker_label"
            )
        return payload


def empty_artifact_manifest() -> dict[str, Any]:
    """Return a compatible empty manifest for legacy results."""

    return {
        "manifest_version": ARTIFACT_MANIFEST_VERSION,
        "stable": [],
        "optional": [],
        "experimental": [],
    }


def normalize_artifact_manifest(manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize a stored optional manifest without requiring clients to read it.

    Missing, malformed, or forward-compatible unknown entries are tolerated for
    reads. Known entry fields are copied only after public-safe normalization,
    so local paths, host details, tokens, job ids, or speaker ids from unknown
    fields cannot leak through this helper.
    """

    if not isinstance(manifest, Mapping):
        return empty_artifact_manifest()

    normalized = empty_artifact_manifest()
    version = manifest.get("manifest_version")
    if isinstance(version, str) and version.strip():
        normalized["manifest_version"] = _public_safe_text(version, "manifest_version")

    for category in ARTIFACT_MANIFEST_CATEGORIES:
        entries = manifest.get(category, [])
        if not isinstance(entries, list):
            continue
        for raw_entry in entries:
            if not isinstance(raw_entry, Mapping):
                continue
            try:
                normalized[category].append(
                    ArtifactManifestEntry.from_mapping(raw_entry).as_dict()
                )
            except ValueError:
                # Legacy/forward-compatible reads tolerate bad optional entries.
                continue
    return normalized


def build_artifact_manifest(
    stable: list[ArtifactManifestEntry],
    optional: list[ArtifactManifestEntry] | None = None,
    experimental: list[ArtifactManifestEntry] | None = None,
) -> dict[str, Any]:
    """Build the optional artifact manifest for a completed transcription."""

    return normalize_artifact_manifest(
        {
            "manifest_version": ARTIFACT_MANIFEST_VERSION,
            "stable": [entry.as_dict() for entry in stable],
            "optional": [entry.as_dict() for entry in optional or []],
            "experimental": [entry.as_dict() for entry in experimental or []],
        }
    )


@runtime_checkable
class TranscriptionArtifactStore(Protocol):
    """Stable slot for persisting completed transcription artifacts."""

    def persist_transcription(
        self, request: TranscriptionArtifactWriteRequest
    ) -> PersistedTranscriptionArtifacts: ...


__all__ = [
    "ARTIFACT_MANIFEST_VERSION",
    "ARTIFACT_MANIFEST_CATEGORIES",
    "AsyncUploadReader",
    "AudioArtifactIndex",
    "ArtifactManifestEntry",
    "PersistedTranscriptionArtifacts",
    "SavedUploadArtifact",
    "TranscriptionArtifactStore",
    "TranscriptionArtifactWriteRequest",
    "UploadPersistenceRequest",
    "build_artifact_manifest",
    "empty_artifact_manifest",
    "normalize_artifact_manifest",
]
