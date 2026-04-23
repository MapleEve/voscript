"""Persistence adapters for completed transcription artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from infra.audio.paths import safe_speaker_label
from infra.job_persistence import _atomic_write_json
from pipeline.contracts import (
    PersistedTranscriptionArtifacts,
    TranscriptionArtifactStore,
    TranscriptionArtifactWriteRequest,
)


class FilesystemTranscriptionArtifactStore(TranscriptionArtifactStore):
    """Persist completed transcription payloads and speaker embeddings to disk."""

    def persist_transcription(
        self, request: TranscriptionArtifactWriteRequest
    ) -> PersistedTranscriptionArtifacts:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = request.output_dir / "result.json"
        _atomic_write_json(
            result_path,
            request.transcription,
            ensure_ascii=False,
            indent=2,
        )

        embedding_paths: dict[str, Path] = {}
        for speaker_label, embedding in request.speaker_embeddings.items():
            safe_label = safe_speaker_label(speaker_label)
            emb_path = request.output_dir / f"emb_{safe_label}.npy"
            np.save(emb_path, embedding)
            embedding_paths[speaker_label] = emb_path

        return PersistedTranscriptionArtifacts(
            result_path=result_path,
            embedding_paths=embedding_paths,
        )


default_transcription_artifact_store = FilesystemTranscriptionArtifactStore()


def persist_transcription_artifacts(
    output_dir: Path,
    transcription: dict,
    speaker_embeddings: dict,
) -> PersistedTranscriptionArtifacts:
    """Persist the completed transcription bundle through the default store."""

    request = TranscriptionArtifactWriteRequest(
        output_dir=output_dir,
        transcription=transcription,
        speaker_embeddings=speaker_embeddings,
    )
    return default_transcription_artifact_store.persist_transcription(request)


__all__ = [
    "FilesystemTranscriptionArtifactStore",
    "default_transcription_artifact_store",
    "persist_transcription_artifacts",
]
