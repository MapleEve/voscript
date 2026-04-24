"""Provider entrypoints for the punc step."""

from __future__ import annotations

from typing import cast

from pipeline.contracts import PipelineContext
from pipeline.registry import resolve_provider

from .default import DefaultPunctuationProvider, default_punc_provider


def run_punc(context: PipelineContext, provider_name: str = "default") -> None:
    """Apply the selected punctuation provider to the shared context."""

    provider = cast(DefaultPunctuationProvider, resolve_provider("punc", provider_name))
    provider.run(context)


__all__ = [
    "DefaultPunctuationProvider",
    "default_punc_provider",
    "run_punc",
]
