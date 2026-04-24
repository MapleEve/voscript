"""Provider entrypoints for the ingest step."""

from __future__ import annotations

from typing import cast

from pipeline.contracts import PipelineContext
from pipeline.registry import resolve_provider

from .default import DefaultIngestProvider, default_ingest_provider


def run_ingest(context: PipelineContext, provider_name: str = "default") -> None:
    """Apply the selected ingest provider to the shared pipeline context."""

    provider = cast(DefaultIngestProvider, resolve_provider("ingest", provider_name))
    provider.run(context)


__all__ = [
    "DefaultIngestProvider",
    "default_ingest_provider",
    "run_ingest",
]
