"""Provider entrypoints for the speaker embedding step."""

from __future__ import annotations

from typing import Any, cast

from pipeline.contracts import (
    SpeakerEmbeddingProvider,
    SpeakerEmbeddingRequest,
    SpeakerEmbeddingResult,
)
from pipeline.registry import resolve_provider

from .default import (
    PipelineMethodSpeakerEmbeddingProvider,
    default_speaker_embedding_provider,
    extract_embeddings_for_turns,
)


def extract_speaker_embeddings(
    pipeline: Any,
    audio_path: str,
    diarization_turns: list[dict[str, Any]],
    provider_name: str = "default",
) -> SpeakerEmbeddingResult:
    """Compatibility helper around the selected embedding provider."""

    provider = cast(SpeakerEmbeddingProvider, resolve_provider("embedding", provider_name))
    request = SpeakerEmbeddingRequest(
        pipeline=pipeline,
        audio_path=audio_path,
        diarization_turns=diarization_turns,
    )
    return provider.extract_embeddings(request)


__all__ = [
    "PipelineMethodSpeakerEmbeddingProvider",
    "default_speaker_embedding_provider",
    "extract_speaker_embeddings",
    "extract_embeddings_for_turns",
]
