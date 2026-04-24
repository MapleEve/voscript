"""Stable slot for pipeline input ingestion and handoff."""

from __future__ import annotations

from typing import TYPE_CHECKING

from providers.ingest import run_ingest

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Seed the pipeline context through the selected ingest provider."""

    run_ingest(
        context,
        provider_name=context.request.provider_for("ingest"),
    )
