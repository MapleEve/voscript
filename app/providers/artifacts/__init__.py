"""Provider entrypoints for the artifacts step."""

from __future__ import annotations

from typing import cast

from pipeline.contracts import PipelineContext, PipelineResult
from pipeline.registry import resolve_provider

from .default import InMemoryArtifactsProvider, default_artifacts_provider


def build_pipeline_artifacts(
    context: PipelineContext, provider_name: str = "default"
) -> PipelineResult:
    """Build the current in-memory artifact bundle through the provider boundary."""

    provider = cast(
        InMemoryArtifactsProvider,
        resolve_provider("artifacts", provider_name),
    )
    return provider.build(context)


__all__ = [
    "InMemoryArtifactsProvider",
    "build_pipeline_artifacts",
    "default_artifacts_provider",
]
