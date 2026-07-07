"""Stable slot for punctuation restoration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.registry import resolve_provider

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Keep punctuation as an explicit slot for later model substitution."""

    provider = resolve_provider("punc", context.request.provider_for("punc"))
    provider.run(context)
