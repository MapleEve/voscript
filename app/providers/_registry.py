"""Compatibility guard for direct provider facade helpers.

Provider selection is owned by :mod:`pipeline.registry`. The ``providers.*``
facade helpers remain only for legacy direct calls and must not route named
provider selection back through the pipeline registry.
"""

from __future__ import annotations

DEFAULT_PROVIDER_NAME = "default"


class ProviderFacadeSelectionError(ValueError):
    """Raised when a direct provider facade is asked to select a provider."""


def require_default_provider(
    step: str,
    provider_name: str = DEFAULT_PROVIDER_NAME,
) -> None:
    """Reject named provider selection from legacy direct provider facades."""

    if provider_name != DEFAULT_PROVIDER_NAME:
        raise ProviderFacadeSelectionError(
            f"providers.{step} facade helpers only support "
            f"provider_name={DEFAULT_PROVIDER_NAME!r}; use pipeline.registry or "
            "PipelineRunner for named provider selection."
        )


__all__ = [
    "DEFAULT_PROVIDER_NAME",
    "ProviderFacadeSelectionError",
    "require_default_provider",
]
