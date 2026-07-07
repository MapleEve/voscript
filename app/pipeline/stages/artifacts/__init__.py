"""Stable slot for final pipeline result assembly."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.registry import resolve_provider

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Emit the current in-memory pipeline artifact bundle."""

    provider = resolve_provider("artifacts", context.request.provider_for("artifacts"))
    context.result = provider.build(context)
    result = context.result.as_dict() if hasattr(context.result, "as_dict") else {}
    context.metadata["artifacts"] = {
        "segment_count": len(result.get("segments", context.aligned_segments)),
        "speaker_count": len(result.get("unique_speakers", context.speaker_embeddings)),
        "persisted": bool(result.get("artifact_paths")),
    }
