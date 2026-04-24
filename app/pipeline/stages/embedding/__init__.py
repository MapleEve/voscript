"""Stable slot for speaker embedding extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from providers.embedding import extract_speaker_embeddings

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Extract speaker embeddings after diarization has defined the turns."""

    result = extract_speaker_embeddings(
        context.pipeline,
        context.embedding_audio_path,
        context.diarization_turns,
        provider_name=context.request.provider_for("embedding"),
    )
    context.speaker_embeddings = result.speaker_embeddings
    context.metadata["embedding"] = {
        "speaker_count": len(result.speaker_embeddings),
    }
