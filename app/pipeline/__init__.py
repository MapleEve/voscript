"""Stable public interface for the transcription pipeline package."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contracts import PipelineContext, PipelineRequest
    from .orchestrator import TranscriptionPipeline
    from .runner import PipelineRunner

__all__ = [
    "PipelineContext",
    "PipelineRequest",
    "PipelineRunner",
    "TranscriptionPipeline",
]


def __getattr__(name: str):
    if name == "TranscriptionPipeline":
        from .orchestrator import TranscriptionPipeline

        return TranscriptionPipeline
    if name == "PipelineContext":
        from .contracts import PipelineContext

        return PipelineContext
    if name == "PipelineRunner":
        from .runner import PipelineRunner

        return PipelineRunner
    if name == "PipelineRequest":
        from .contracts import PipelineRequest

        return PipelineRequest
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
