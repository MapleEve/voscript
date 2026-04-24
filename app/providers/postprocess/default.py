"""Default provider for transcript post-processing."""

from __future__ import annotations

from pipeline.contracts import PipelineContext


class DefaultPostprocessProvider:
    """Keep post-processing as a stable no-op boundary."""

    def run(self, context: PipelineContext) -> None:
        context.metadata["postprocess"] = {
            "status": "pass_through",
        }


default_postprocess_provider = DefaultPostprocessProvider()


__all__ = [
    "DefaultPostprocessProvider",
    "default_postprocess_provider",
]
