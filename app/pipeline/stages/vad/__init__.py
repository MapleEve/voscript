"""Stable slot for voice activity detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from providers.vad import run_vad

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Capture VAD policy through the selected stable provider."""

    run_vad(
        context,
        provider_name=context.request.provider_for("vad"),
    )
