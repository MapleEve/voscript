"""Provider entrypoints for the vad step."""

from __future__ import annotations

from pipeline.contracts import PipelineContext
from providers._registry import require_default_provider

from .default import DefaultVADProvider, default_vad_provider


def run_vad(context: PipelineContext, provider_name: str = "default") -> None:
    """Apply the default VAD provider to the shared pipeline context."""

    require_default_provider("vad", provider_name)
    default_vad_provider.run(context)


__all__ = [
    "DefaultVADProvider",
    "default_vad_provider",
    "run_vad",
]
