"""Shared errors for pipeline stage and provider resolution."""

from __future__ import annotations


class PipelineLookupError(LookupError):
    """Base class for stable pipeline registry lookup failures."""


class StageNotFoundError(PipelineLookupError):
    """Raised when a stable pipeline stage slot cannot be resolved."""


class ProviderNotFoundError(PipelineLookupError):
    """Raised when a pipeline provider implementation cannot be resolved."""


__all__ = [
    "PipelineLookupError",
    "ProviderNotFoundError",
    "StageNotFoundError",
]
