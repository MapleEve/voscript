"""Stable slot for transcript post-processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.registry import resolve_provider

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Reserve a stable boundary for LLM or rule-based transcript cleanup."""

    provider = resolve_provider(
        "postprocess",
        context.request.provider_for("postprocess"),
    )
    provider.run(context)
