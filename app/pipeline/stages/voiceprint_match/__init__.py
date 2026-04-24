"""Stable slot for voiceprint matching and AS-norm scoring."""

from __future__ import annotations

from typing import TYPE_CHECKING

from providers.voiceprint_match import match_speaker_embeddings

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Run voiceprint matching through the provider boundary when available."""

    if context.request.status_callback is not None:
        context.request.status_callback("identifying")

    result = match_speaker_embeddings(
        context.speaker_embeddings,
        voiceprint_db=context.request.voiceprint_db,
        threshold=context.request.voiceprint_threshold,
        provider_name=context.request.provider_for("voiceprint_match"),
    )
    context.voiceprint_matches = result.speaker_map
    context.metadata["voiceprint_match"] = {
        "applied": result.applied,
        "speaker_count": len(result.speaker_map),
        "reason": result.reason,
    }
    if result.threshold is not None:
        context.metadata["voiceprint_match"]["threshold"] = result.threshold
