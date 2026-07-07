"""Provider entrypoints for the punc step."""

from __future__ import annotations

from pipeline.contracts import PipelineContext
from providers._registry import require_default_provider

from .default import DefaultPunctuationProvider, default_punc_provider


def run_punc(context: PipelineContext, provider_name: str = "default") -> None:
    """Apply the default punctuation provider to the shared context."""

    require_default_provider("punc", provider_name)
    default_punc_provider.run(context)


__all__ = [
    "DefaultPunctuationProvider",
    "default_punc_provider",
    "run_punc",
]
