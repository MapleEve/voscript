"""Stable contracts for voiceprint matching providers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class VoiceprintMatchRequest:
    """Describe the embeddings to match plus the optional backing DB."""

    speaker_embeddings: dict[str, Any]
    voiceprint_db: Any | None = None
    threshold: float | None = None


@dataclass(frozen=True, slots=True)
class VoiceprintMatchResult:
    """Wrap the speaker-map style payload emitted by a matcher."""

    speaker_map: dict[str, dict[str, Any]]
    applied: bool
    threshold: float | None = None
    reason: str | None = None


@runtime_checkable
class VoiceprintMatchProvider(Protocol):
    """Canonical slot for AS-norm voiceprint matching implementations."""

    def match(self, request: VoiceprintMatchRequest) -> VoiceprintMatchResult: ...
