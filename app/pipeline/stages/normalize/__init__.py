"""Stable slot for in-pipeline audio normalization."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.contracts import AudioNormalizationRequest
from pipeline.registry import resolve_provider

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Normalize the ingest path into the canonical WAV shape for later stages."""

    if context.request.status_callback is not None:
        context.request.status_callback("converting")

    input_path = Path(context.working_audio_path or context.request.audio_path)
    provider = resolve_provider("normalize", context.request.provider_for("normalize"))
    result = provider.normalize(
        AudioNormalizationRequest(input_path=input_path)
    )

    context.working_audio_path = str(result.normalized_path)
    if context.request.raw_audio_path is None:
        context.embedding_audio_path = str(result.normalized_path)
    if not result.reused_source and result.normalized_path != result.source_path:
        context.temporary_paths.append(result.normalized_path)

    context.metadata["normalize"] = {
        "status": "completed",
        "source_path": str(result.source_path),
        "working_audio_path": context.working_audio_path,
        "reused_source": result.reused_source,
    }
