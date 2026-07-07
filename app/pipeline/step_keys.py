"""Canonical keys for stable pipeline stages and provider selectors."""

from __future__ import annotations

_STEP_ALIASES = {
    "input_normalization": "normalize",
    "enhancement": "enhance",
}


def normalize_token(value: str, *, field_name: str) -> str:
    token = value.strip().lower().replace("-", "_")
    if not token:
        raise ValueError(f"{field_name} must not be empty")
    return token


def canonical_step_name(step: str) -> str:
    """Map compatibility aliases onto the canonical stable step names."""

    token = normalize_token(step, field_name="step")
    return _STEP_ALIASES.get(token, token)


__all__ = [
    "canonical_step_name",
    "normalize_token",
]
