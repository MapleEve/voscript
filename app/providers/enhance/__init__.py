"""Provider entrypoints for the enhance step."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pipeline.contracts import (
    AudioEnhancementProvider,
    AudioEnhancementRequest,
    AudioEnhancementResult,
)
from pipeline.registry import resolve_provider

from .default import (
    ConditionalDenoiseEnhancer,
    default_audio_enhancer,
    default_enhance_provider,
)


def enhance_audio(
    wav_path: Path,
    model: str | None = None,
    snr_threshold: float | None = None,
    provider_name: str = "default",
) -> AudioEnhancementResult:
    """Run the selected enhancement provider and return the full contract result."""

    provider = cast(
        AudioEnhancementProvider,
        resolve_provider("enhance", provider_name),
    )
    request = AudioEnhancementRequest(
        wav_path=wav_path,
        model=model,
        snr_threshold=snr_threshold,
    )
    return provider.enhance(request)


def maybe_denoise(
    wav_path: Path,
    model: str | None = None,
    snr_threshold: float | None = None,
    provider_name: str = "default",
) -> Path:
    """Compatibility helper around the selected enhance provider."""

    return enhance_audio(
        wav_path,
        model=model,
        snr_threshold=snr_threshold,
        provider_name=provider_name,
    ).output_path


__all__ = [
    "ConditionalDenoiseEnhancer",
    "default_audio_enhancer",
    "default_enhance_provider",
    "enhance_audio",
    "maybe_denoise",
]
