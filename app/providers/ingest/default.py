"""Default provider for seeding pipeline input paths."""

from __future__ import annotations

from pipeline.contracts import PipelineContext


class DefaultIngestProvider:
    """Seed the execution context with externally prepared input paths."""

    def run(self, context: PipelineContext) -> None:
        context.working_audio_path = context.request.audio_path
        context.embedding_audio_path = (
            context.request.raw_audio_path or context.request.audio_path
        )
        context.metadata["ingest"] = {
            "working_audio_path": context.working_audio_path,
            "embedding_audio_path": context.embedding_audio_path,
        }


default_ingest_provider = DefaultIngestProvider()


__all__ = [
    "DefaultIngestProvider",
    "default_ingest_provider",
]
