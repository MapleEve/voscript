"""Provider entrypoints for the enhance step."""

from __future__ import annotations

from pathlib import Path

from pipeline.contracts import (
    AudioEnhancementProvider,
    AudioEnhancementRequest,
    AudioEnhancementResult,
)
from providers._registry import require_default_provider

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
    """Run the default enhancement provider and return the full contract result."""

    require_default_provider("enhance", provider_name)
    provider: AudioEnhancementProvider = default_enhance_provider
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
    """Compatibility helper around the default enhance provider."""

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
