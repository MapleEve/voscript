"""Stable pipeline stage entrypoints."""

from __future__ import annotations

from pipeline.registry import available_stage_slots, resolve_stage

__all__ = [
    "available_stage_slots",
    "resolve_stage",
]
