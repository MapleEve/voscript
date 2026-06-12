"""Compatibility re-exports for diarization alignment helpers."""

from __future__ import annotations

from postprocess.alignment import (
    assign_segment_speaker,
    build_aligned_segments,
    dedup_short_segments,
    normalize_segment,
    normalize_words,
)

__all__ = [
    "assign_segment_speaker",
    "build_aligned_segments",
    "dedup_short_segments",
    "normalize_segment",
    "normalize_words",
]
