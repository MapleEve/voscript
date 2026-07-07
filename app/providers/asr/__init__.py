"""Provider entrypoints for the ASR step."""

from __future__ import annotations

from typing import Any

from pipeline.contracts import ASRProvider, ASRRequest, ASRResult
from providers._registry import require_default_provider

from .default import PipelineMethodASRProvider, default_asr_provider


def transcribe_audio(
    pipeline: Any,
    audio_path: str,
    language: str | None = None,
    no_repeat_ngram_size: int | None = None,
    provider_name: str = "default",
) -> ASRResult:
    """Compatibility helper around the default ASR provider."""

    require_default_provider("asr", provider_name)
    provider: ASRProvider = default_asr_provider
    request = ASRRequest(
        pipeline=pipeline,
        audio_path=audio_path,
        language=language,
        no_repeat_ngram_size=no_repeat_ngram_size,
    )
    return provider.transcribe(request)


__all__ = [
    "PipelineMethodASRProvider",
    "default_asr_provider",
    "transcribe_audio",
]
