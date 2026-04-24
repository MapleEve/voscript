"""Stable contracts for automatic speech recognition providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ASRRequest:
    """Describe the audio asset and decode controls for an ASR run."""

    pipeline: Any
    audio_path: str
    language: str | None = None
    no_repeat_ngram_size: int | None = None


@dataclass(frozen=True, slots=True)
class ASRResult:
    """Wrap the transcription payload emitted by an ASR provider."""

    transcription_result: dict[str, Any]


@runtime_checkable
class ASRProvider(Protocol):
    """Canonical slot for pluggable ASR implementations."""

    def transcribe(self, request: ASRRequest) -> ASRResult: ...
