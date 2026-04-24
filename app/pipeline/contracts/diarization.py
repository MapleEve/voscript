"""Stable contracts for diarization and alignment providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class DiarizationRequest:
    """Describe the inputs required to run diarization and segment alignment."""

    pipeline: Any
    audio_path: str
    transcription_result: dict[str, Any]
    min_speakers: int | None = None
    max_speakers: int | None = None


@dataclass(frozen=True, slots=True)
class DiarizationResult:
    """Wrap diarization turns and aligned segments emitted by a provider."""

    turns: list[dict[str, Any]]
    aligned_segments: list[dict[str, Any]]
    dedup_removed: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class DiarizationProvider(Protocol):
    """Canonical slot for diarization and overlap-handling implementations."""

    def diarize(self, request: DiarizationRequest) -> DiarizationResult: ...
