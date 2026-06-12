"""Lazy compatibility wrappers around the pipeline registry."""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _registry() -> Any:
    return import_module("pipeline.registry")


def available_providers(step: str) -> tuple[str, ...]:
    return _registry().available_providers(step)


def available_stage_slots() -> tuple[str, ...]:
    return _registry().available_stage_slots()


def register_provider(step: str, name: str, provider: Any) -> None:
    _registry().register_provider(step, name, provider)


def resolve_provider(step: str, name: str = "default") -> Any:
    return _registry().resolve_provider(step, name)


def unregister_provider(step: str, name: str) -> None:
    _registry().unregister_provider(step, name)


__all__ = [
    "available_providers",
    "available_stage_slots",
    "register_provider",
    "resolve_provider",
    "unregister_provider",
]
