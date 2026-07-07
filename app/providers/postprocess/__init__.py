"""Provider entrypoints for the postprocess step."""

from __future__ import annotations

from pipeline.contracts import PipelineContext
from providers._registry import require_default_provider

from .default import DefaultPostprocessProvider, default_postprocess_provider


def run_postprocess(context: PipelineContext, provider_name: str = "default") -> None:
    """Apply the default post-process provider to the shared context."""

    require_default_provider("postprocess", provider_name)
    default_postprocess_provider.run(context)


__all__ = [
    "DefaultPostprocessProvider",
    "default_postprocess_provider",
    "run_postprocess",
]
