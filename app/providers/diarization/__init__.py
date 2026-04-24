"""Provider entrypoints for the diarization step."""

from __future__ import annotations

from typing import Any, cast

from pipeline.contracts import (
    DiarizationProvider,
    DiarizationRequest,
    DiarizationResult,
)
from pipeline.registry import resolve_provider

from .default import PipelineMethodDiarizationProvider, default_diarization_provider
from .default import align_diarized_segments, run_pyannote_diarization


def run_diarization(
    pipeline: Any,
    audio_path: str,
    transcription_result: dict[str, Any],
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    provider_name: str = "default",
) -> DiarizationResult:
    """Compatibility helper around the selected diarization provider."""

    provider = cast(DiarizationProvider, resolve_provider("diarization", provider_name))
    request = DiarizationRequest(
        pipeline=pipeline,
        audio_path=audio_path,
        transcription_result=transcription_result,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )
    return provider.diarize(request)


__all__ = [
    "PipelineMethodDiarizationProvider",
    "align_diarized_segments",
    "default_diarization_provider",
    "run_pyannote_diarization",
    "run_diarization",
]
