"""Stable contracts for input-audio normalization providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class AudioNormalizationRequest:
    """Describe the source asset and target canonical audio format."""

    input_path: Path
    target_sample_rate: int = 16000
    target_channels: int = 1
    target_format: str = "wav"


@dataclass(frozen=True, slots=True)
class AudioNormalizationResult:
    """Describe the normalized output selected or produced by a provider."""

    source_path: Path
    normalized_path: Path
    reused_source: bool


@runtime_checkable
class InputNormalizationProvider(Protocol):
    """Canonical slot for converting uploads into pipeline-ready audio."""

    def normalize(
        self, request: AudioNormalizationRequest
    ) -> AudioNormalizationResult: ...


__all__ = [
    "AudioNormalizationRequest",
    "AudioNormalizationResult",
    "InputNormalizationProvider",
]
