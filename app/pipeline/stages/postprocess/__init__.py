"""Stable slot for transcript post-processing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from providers.postprocess import run_postprocess

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Reserve a stable boundary for LLM or rule-based transcript cleanup."""

    run_postprocess(
        context,
        provider_name=context.request.provider_for("postprocess"),
    )
