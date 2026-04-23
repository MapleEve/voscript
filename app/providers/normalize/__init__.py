"""Provider entrypoints for the normalize step."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pipeline.contracts import (
    AudioNormalizationRequest,
    AudioNormalizationResult,
    InputNormalizationProvider,
)
from pipeline.registry import resolve_provider

from .default import (
    FFmpegInputNormalizer,
    default_input_normalizer,
    default_normalize_provider,
)


def normalize_audio(
    input_path: Path, provider_name: str = "default"
) -> AudioNormalizationResult:
    """Run the selected normalize provider and return the full contract result."""

    provider = cast(
        InputNormalizationProvider,
        resolve_provider("normalize", provider_name),
    )
    request = AudioNormalizationRequest(input_path=input_path)
    return provider.normalize(request)


def convert_to_wav(input_path: Path, provider_name: str = "default") -> Path:
    """Compatibility helper around the selected normalize provider."""

    return normalize_audio(input_path, provider_name=provider_name).normalized_path


__all__ = [
    "FFmpegInputNormalizer",
    "convert_to_wav",
    "default_input_normalizer",
    "default_normalize_provider",
    "normalize_audio",
]
