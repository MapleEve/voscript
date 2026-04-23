"""Shared request objects for stable pipeline execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any


def _normalize_provider_name(name: str) -> str:
    token = name.strip().lower().replace("-", "_")
    if not token:
        raise ValueError("provider name must not be empty")
    return token


@dataclass(frozen=True, slots=True)
class PipelineRequest:
    """Transport the current pipeline invocation parameters."""

    audio_path: str
    raw_audio_path: str | None = None
    language: str | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None
    no_repeat_ngram_size: int | None = None
    voiceprint_db: Any | None = None
    voiceprint_threshold: float | None = None
    denoise_model: str | None = None
    snr_threshold: float | None = None
    artifact_dir: Path | None = None
    status_callback: Callable[[str], None] | None = None
    provider_selection: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: dict[str, str] = {}
        if self.provider_selection:
            from pipeline.registry import canonical_step_name

            for step_name, provider_name in self.provider_selection.items():
                step_key = canonical_step_name(str(step_name))
                normalized[step_key] = _normalize_provider_name(str(provider_name))
        object.__setattr__(
            self,
            "provider_selection",
            MappingProxyType(normalized),
        )

    def provider_for(self, step: str, default: str = "default") -> str:
        """Return the explicitly selected provider for a step, or the fallback."""

        from pipeline.registry import canonical_step_name

        step_key = canonical_step_name(step)
        return self.provider_selection.get(step_key, _normalize_provider_name(default))


__all__ = ["PipelineRequest"]
