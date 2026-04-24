"""Provider entrypoints for the postprocess step."""

from __future__ import annotations

from typing import cast

from pipeline.contracts import PipelineContext
from pipeline.registry import resolve_provider

from .default import DefaultPostprocessProvider, default_postprocess_provider


def run_postprocess(context: PipelineContext, provider_name: str = "default") -> None:
    """Apply the selected post-process provider to the shared context."""

    provider = cast(
        DefaultPostprocessProvider,
        resolve_provider("postprocess", provider_name),
    )
    provider.run(context)


__all__ = [
    "DefaultPostprocessProvider",
    "default_postprocess_provider",
    "run_postprocess",
]
