"""Provider entrypoints for the voiceprint matching step."""

from __future__ import annotations

from typing import Any, cast

from pipeline.contracts import (
    VoiceprintMatchProvider,
    VoiceprintMatchRequest,
    VoiceprintMatchResult,
)
from pipeline.registry import resolve_provider

from .default import DefaultVoiceprintMatchProvider, default_voiceprint_match_provider


def match_speaker_embeddings(
    speaker_embeddings: dict[str, Any],
    voiceprint_db: Any | None = None,
    threshold: float | None = None,
    provider_name: str = "default",
) -> VoiceprintMatchResult:
    """Compatibility helper around the selected voiceprint matcher."""

    provider = cast(
        VoiceprintMatchProvider,
        resolve_provider("voiceprint_match", provider_name),
    )
    request = VoiceprintMatchRequest(
        speaker_embeddings=speaker_embeddings,
        voiceprint_db=voiceprint_db,
        threshold=threshold,
    )
    return provider.match(request)


__all__ = [
    "DefaultVoiceprintMatchProvider",
    "default_voiceprint_match_provider",
    "match_speaker_embeddings",
]
