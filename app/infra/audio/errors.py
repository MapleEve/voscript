"""Typed errors for audio filesystem helpers."""

from __future__ import annotations


class AudioPathError(ValueError):
    """Base error for invalid audio filesystem inputs."""


class InvalidTranscriptionIdError(AudioPathError):
    """Raised when a transcription ID cannot be safely used as a path segment."""


class AudioPathTraversalError(AudioPathError):
    """Raised when a resolved audio path escapes its configured root."""


class InvalidSpeakerLabelError(AudioPathError):
    """Raised when a speaker label cannot be safely used in a filename."""


__all__ = [
    "AudioPathError",
    "AudioPathTraversalError",
    "InvalidSpeakerLabelError",
    "InvalidTranscriptionIdError",
]
