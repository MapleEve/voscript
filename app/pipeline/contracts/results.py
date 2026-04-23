"""Shared result objects for stable pipeline execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Canonical in-memory result bundle emitted by the pipeline."""

    segments: list[dict[str, Any]]
    speaker_embeddings: dict[str, Any]
    unique_speakers: list[str]
    transcription: dict[str, Any] | None = None
    artifact_paths: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-serialisable payload exposed by current callers."""

        payload = {
            "segments": self.segments,
            "speaker_embeddings": self.speaker_embeddings,
            "unique_speakers": self.unique_speakers,
        }
        if self.transcription is not None:
            payload["transcription"] = self.transcription
        if self.artifact_paths is not None:
            payload["artifact_paths"] = self.artifact_paths
        return payload


__all__ = ["PipelineResult"]
