"""Provider entrypoints for the vad step."""

from __future__ import annotations

from typing import cast

from pipeline.contracts import PipelineContext
from pipeline.registry import resolve_provider

from .default import DefaultVADProvider, default_vad_provider


def run_vad(context: PipelineContext, provider_name: str = "default") -> None:
    """Apply the selected VAD provider to the shared pipeline context."""

    provider = cast(DefaultVADProvider, resolve_provider("vad", provider_name))
    provider.run(context)


__all__ = [
    "DefaultVADProvider",
    "default_vad_provider",
    "run_vad",
]
