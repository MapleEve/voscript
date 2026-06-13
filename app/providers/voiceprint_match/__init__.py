"""Provider entrypoints for the voiceprint matching step."""

from __future__ import annotations

from typing import Any

from pipeline.contracts import (
    VoiceprintMatchProvider,
    VoiceprintMatchRequest,
    VoiceprintMatchResult,
)
from providers._registry import require_default_provider

from .default import DefaultVoiceprintMatchProvider, default_voiceprint_match_provider


def match_speaker_embeddings(
    speaker_embeddings: dict[str, Any],
    voiceprint_db: Any | None = None,
    threshold: float | None = None,
    provider_name: str = "default",
) -> VoiceprintMatchResult:
    """Compatibility helper around the default voiceprint matcher."""

    require_default_provider("voiceprint_match", provider_name)
    provider: VoiceprintMatchProvider = default_voiceprint_match_provider
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
