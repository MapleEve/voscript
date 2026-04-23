"""Shared mutable context passed across stable pipeline stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .requests import PipelineRequest
from .results import PipelineResult


@dataclass(slots=True)
class PipelineContext:
    """Mutable execution state shared across stable pipeline stages."""

    pipeline: Any
    request: PipelineRequest
    working_audio_path: str = ""
    embedding_audio_path: str = ""
    transcription_result: dict[str, Any] | None = None
    diarization_turns: list[dict[str, Any]] = field(default_factory=list)
    aligned_segments: list[dict[str, Any]] = field(default_factory=list)
    speaker_embeddings: dict[str, Any] = field(default_factory=dict)
    voiceprint_matches: dict[str, Any] = field(default_factory=dict)
    temporary_paths: list[Path] = field(default_factory=list)
    result: PipelineResult | dict[str, Any] | None = None
    metadata: dict[str, Any] = field(
        default_factory=lambda: {"executed_stages": []}
    )

    def mark_stage(self, stage_name: str) -> None:
        self.metadata.setdefault("executed_stages", []).append(stage_name)

    def to_result(self) -> dict[str, Any]:
        if isinstance(self.result, PipelineResult):
            return self.result.as_dict()
        if self.result is not None:
            return self.result
        return PipelineResult(
            segments=self.aligned_segments,
            speaker_embeddings=self.speaker_embeddings,
            unique_speakers=list(self.speaker_embeddings.keys()),
        ).as_dict()


__all__ = ["PipelineContext"]
