"""Stable contracts for speaker embedding providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class SpeakerEmbeddingRequest:
    """Describe the diarized turns that should be embedded."""

    pipeline: Any
    audio_path: str
    diarization_turns: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class SpeakerEmbeddingResult:
    """Wrap the embeddings emitted for each diarization cluster."""

    speaker_embeddings: dict[str, Any]


@runtime_checkable
class SpeakerEmbeddingProvider(Protocol):
    """Canonical slot for pluggable speaker embedding implementations."""

    def extract_embeddings(
        self, request: SpeakerEmbeddingRequest
    ) -> SpeakerEmbeddingResult: ...
