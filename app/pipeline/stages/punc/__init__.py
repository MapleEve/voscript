"""Stable slot for punctuation restoration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from providers.punc import run_punc

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Keep punctuation as an explicit slot for later model substitution."""

    run_punc(
        context,
        provider_name=context.request.provider_for("punc"),
    )
