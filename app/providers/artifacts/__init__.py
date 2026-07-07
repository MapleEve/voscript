"""Provider entrypoints for the artifacts step."""

from __future__ import annotations

from pipeline.contracts import PipelineContext, PipelineResult
from providers._registry import require_default_provider

from .default import InMemoryArtifactsProvider, default_artifacts_provider


def build_pipeline_artifacts(
    context: PipelineContext, provider_name: str = "default"
) -> PipelineResult:
    """Build the current in-memory artifact bundle through the provider boundary."""

    require_default_provider("artifacts", provider_name)
    return default_artifacts_provider.build(context)


__all__ = [
    "InMemoryArtifactsProvider",
    "build_pipeline_artifacts",
    "default_artifacts_provider",
]
