"""Default provider for punctuation handling."""

from __future__ import annotations

from pipeline.contracts import PipelineContext


class DefaultPunctuationProvider:
    """Keep punctuation restoration as a stable pass-through boundary."""

    def run(self, context: PipelineContext) -> None:
        context.metadata["punc"] = {
            "status": "pass_through",
            "reason": "current_asr_output_already_contains_punctuation",
        }


default_punc_provider = DefaultPunctuationProvider()


__all__ = [
    "DefaultPunctuationProvider",
    "default_punc_provider",
]
