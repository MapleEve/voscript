"""Stable slot for diarization and overlap-oriented alignment handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.contracts import (
    DiarizationRequest,
    normalize_public_alignment_metadata,
)
from pipeline.registry import resolve_provider

from .alignment import (
    assign_segment_speaker,
    build_aligned_segments,
    dedup_short_segments,
    normalize_segment,
    normalize_words,
)

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Run diarization, attach speakers, and apply current overlap cleanup."""

    if context.transcription_result is None:
        raise RuntimeError("ASR stage must run before diarization")

    provider = resolve_provider(
        "diarization",
        context.request.provider_for("diarization"),
    )
    result = provider.diarize(
        DiarizationRequest(
            pipeline=context.pipeline,
            audio_path=context.working_audio_path,
            transcription_result=context.transcription_result,
            min_speakers=context.request.min_speakers,
            max_speakers=context.request.max_speakers,
        )
    )
    context.diarization_turns = result.turns
    context.aligned_segments = result.aligned_segments
    diarization_metadata = {
        "turn_count": len(result.turns),
        "dedup_removed": result.dedup_removed,
    }
    alignment_metadata = normalize_public_alignment_metadata(
        result.metadata.get("alignment") if result.metadata else None
    )
    if alignment_metadata:
        diarization_metadata["alignment"] = alignment_metadata
    context.metadata["diarization"] = diarization_metadata


__all__ = [
    "assign_segment_speaker",
    "build_aligned_segments",
    "dedup_short_segments",
    "normalize_segment",
    "normalize_words",
    "run",
]
