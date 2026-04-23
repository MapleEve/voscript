"""Default provider for voice activity detection policy."""

from __future__ import annotations

from pipeline.contracts import PipelineContext


class DefaultVADProvider:
    """Record the current VAD policy while ASR still owns execution."""

    def run(self, context: PipelineContext) -> None:
        context.metadata["vad"] = {
            "status": "embedded_in_asr",
            "backend": "faster_whisper_vad_filter",
        }


default_vad_provider = DefaultVADProvider()


__all__ = [
    "DefaultVADProvider",
    "default_vad_provider",
]
