"""Stable slot for speaker embedding extraction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.contracts import SpeakerEmbeddingRequest
from pipeline.registry import resolve_provider

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Extract speaker embeddings after diarization has defined the turns."""

    provider = resolve_provider("embedding", context.request.provider_for("embedding"))
    result = provider.extract_embeddings(
        SpeakerEmbeddingRequest(
            pipeline=context.pipeline,
            audio_path=context.embedding_audio_path,
            diarization_turns=context.diarization_turns,
        )
    )
    context.speaker_embeddings = result.speaker_embeddings
    context.metadata["embedding"] = {
        "speaker_count": len(result.speaker_embeddings),
    }
