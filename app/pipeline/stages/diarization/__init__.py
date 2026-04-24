"""Stable slot for diarization and overlap-oriented alignment handling."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    from providers.diarization import run_diarization

    if context.transcription_result is None:
        raise RuntimeError("ASR stage must run before diarization")

    result = run_diarization(
        context.pipeline,
        context.working_audio_path,
        context.transcription_result,
        min_speakers=context.request.min_speakers,
        max_speakers=context.request.max_speakers,
        provider_name=context.request.provider_for("diarization"),
    )
    context.diarization_turns = result.turns
    context.aligned_segments = result.aligned_segments
    context.metadata["diarization"] = {
        "turn_count": len(result.turns),
        "dedup_removed": result.dedup_removed,
    }
    if result.metadata:
        context.metadata["diarization"].update(result.metadata)


__all__ = [
    "assign_segment_speaker",
    "build_aligned_segments",
    "dedup_short_segments",
    "normalize_segment",
    "normalize_words",
    "run",
]
