"""Stable slot for final pipeline result assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from providers.artifacts import build_pipeline_artifacts

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Emit the current in-memory pipeline artifact bundle."""

    context.result = build_pipeline_artifacts(
        context,
        provider_name=context.request.provider_for("artifacts"),
    )
    result = context.result.as_dict() if hasattr(context.result, "as_dict") else {}
    context.metadata["artifacts"] = {
        "segment_count": len(result.get("segments", context.aligned_segments)),
        "speaker_count": len(result.get("unique_speakers", context.speaker_embeddings)),
        "persisted": bool(result.get("artifact_paths")),
    }
