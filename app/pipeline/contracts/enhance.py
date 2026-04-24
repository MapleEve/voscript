"""Stable contracts for audio enhancement providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class AudioEnhancementRequest:
    """Describe the normalized WAV and the policy used to enhance it."""

    wav_path: Path
    model: str | None = None
    snr_threshold: float | None = None


@dataclass(frozen=True, slots=True)
class AudioEnhancementResult:
    """Describe the output emitted by an enhancement provider."""

    input_path: Path
    output_path: Path
    applied: bool
    model: str


@runtime_checkable
class AudioEnhancementProvider(Protocol):
    """Canonical slot for denoising and other signal enhancement steps."""

    def enhance(self, request: AudioEnhancementRequest) -> AudioEnhancementResult: ...


__all__ = [
    "AudioEnhancementProvider",
    "AudioEnhancementRequest",
    "AudioEnhancementResult",
]
