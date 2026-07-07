"""Provider entrypoints for the ingest step."""

from __future__ import annotations

from pipeline.contracts import PipelineContext
from providers._registry import require_default_provider

from .default import DefaultIngestProvider, default_ingest_provider


def run_ingest(context: PipelineContext, provider_name: str = "default") -> None:
    """Apply the default ingest provider to the shared pipeline context."""

    require_default_provider("ingest", provider_name)
    default_ingest_provider.run(context)


__all__ = [
    "DefaultIngestProvider",
    "default_ingest_provider",
    "run_ingest",
]
