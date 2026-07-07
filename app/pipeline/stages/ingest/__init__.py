"""Stable slot for pipeline input ingestion and handoff."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.registry import resolve_provider

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Seed the pipeline context through the selected ingest provider."""

    provider = resolve_provider("ingest", context.request.provider_for("ingest"))
    provider.run(context)
