"""Stable slot for voice activity detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.registry import resolve_provider

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Capture VAD policy through the selected stable provider."""

    provider = resolve_provider("vad", context.request.provider_for("vad"))
    provider.run(context)
