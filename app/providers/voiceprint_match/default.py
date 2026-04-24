"""Default provider for voiceprint matching and AS-norm lookup."""

from __future__ import annotations

from typing import Any

from pipeline.contracts import (
    VoiceprintMatchProvider,
    VoiceprintMatchRequest,
    VoiceprintMatchResult,
)


class DefaultVoiceprintMatchProvider(VoiceprintMatchProvider):
    """Use the current VoiceprintDB identify() API when available."""

    def match(self, request: VoiceprintMatchRequest) -> VoiceprintMatchResult:
        if not request.speaker_embeddings:
            return VoiceprintMatchResult(
                speaker_map={},
                applied=False,
                threshold=request.threshold,
                reason="no_embeddings",
            )

        if request.voiceprint_db is None:
            return VoiceprintMatchResult(
                speaker_map={},
                applied=False,
                threshold=request.threshold,
                reason="voiceprint_db_unavailable",
            )

        speaker_map: dict[str, dict[str, Any]] = {}
        for speaker_label, embedding in request.speaker_embeddings.items():
            if request.threshold is None:
                matched_id, matched_name, similarity = request.voiceprint_db.identify(
                    embedding
                )
            else:
                matched_id, matched_name, similarity = request.voiceprint_db.identify(
                    embedding,
                    threshold=request.threshold,
                )
            speaker_map[speaker_label] = {
                "matched_id": matched_id,
                "matched_name": matched_name or speaker_label,
                "similarity": round(similarity, 4),
                "embedding_key": speaker_label,
            }

        return VoiceprintMatchResult(
            speaker_map=speaker_map,
            applied=True,
            threshold=request.threshold,
            reason="matched",
        )


default_voiceprint_match_provider = DefaultVoiceprintMatchProvider()


__all__ = [
    "DefaultVoiceprintMatchProvider",
    "default_voiceprint_match_provider",
]
