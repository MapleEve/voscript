"""Static provider capability metadata and language matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pipeline.registry import canonical_step_name

ALL_LANGUAGES = "*"
StageCriticality = Literal["required", "degradable", "optional"]
FailurePolicy = Literal["hard_fail", "skip"]


class ProviderCapabilityError(RuntimeError):
    """Raised when a required provider capability is not satisfied."""


@dataclass(frozen=True)
class ProviderCapability:
    stage: str
    name: str
    supported_languages: frozenset[str]
    disabled_languages: frozenset[str] = frozenset()
    stage_criticality: StageCriticality = "required"
    supports_rust_kernel: bool = False
    failure_policy: FailurePolicy = "hard_fail"


@dataclass(frozen=True)
class CapabilityMatch:
    capability: ProviderCapability
    language: str | None
    should_run: bool
    reason: str
    metadata: dict[str, str]


_DEFAULT_CAPABILITIES: dict[tuple[str, str], ProviderCapability] = {
    ("asr", "default"): ProviderCapability(
        stage="asr",
        name="default",
        supported_languages=frozenset({ALL_LANGUAGES}),
        stage_criticality="required",
        failure_policy="hard_fail",
    ),
    ("alignment", "default"): ProviderCapability(
        stage="alignment",
        name="default",
        supported_languages=frozenset({ALL_LANGUAGES}),
        disabled_languages=frozenset(),
        stage_criticality="degradable",
        failure_policy="skip",
    ),
    ("diarization", "default"): ProviderCapability(
        stage="diarization",
        name="default",
        supported_languages=frozenset({ALL_LANGUAGES}),
        stage_criticality="required",
        failure_policy="hard_fail",
    ),
    ("embedding", "default"): ProviderCapability(
        stage="embedding",
        name="default",
        supported_languages=frozenset({ALL_LANGUAGES}),
        stage_criticality="required",
        failure_policy="hard_fail",
    ),
    ("voiceprint_match", "default"): ProviderCapability(
        stage="voiceprint_match",
        name="default",
        supported_languages=frozenset({ALL_LANGUAGES}),
        stage_criticality="required",
        failure_policy="hard_fail",
    ),
    ("postprocess", "default"): ProviderCapability(
        stage="postprocess",
        name="default",
        supported_languages=frozenset({ALL_LANGUAGES}),
        stage_criticality="required",
        failure_policy="hard_fail",
    ),
}


def _normalize_provider_name(name: str) -> str:
    token = name.strip().lower().replace("-", "_")
    if not token:
        raise ProviderCapabilityError("provider name must not be empty")
    return token


def _normalize_language(language: str | None) -> str | None:
    if language is None:
        return None
    token = language.strip().lower()
    return token or None


def get_provider_capability(stage: str, name: str = "default") -> ProviderCapability:
    """Return static metadata for a provider stage/name pair."""

    stage_key = canonical_step_name(stage)
    name_key = _normalize_provider_name(name)
    try:
        return _DEFAULT_CAPABILITIES[(stage_key, name_key)]
    except KeyError as exc:
        raise ProviderCapabilityError(
            f"No provider capability registered for stage={stage_key!r} "
            f"name={name_key!r}"
        ) from exc


def _language_supported(capability: ProviderCapability, language: str | None) -> bool:
    if language is None:
        return True
    if language in capability.disabled_languages:
        return False
    return (
        ALL_LANGUAGES in capability.supported_languages
        or language in capability.supported_languages
    )


def match_provider_capability(
    stage: str,
    name: str = "default",
    language: str | None = None,
) -> CapabilityMatch:
    """Match language against provider capability without loading models."""

    capability = get_provider_capability(stage, name)
    normalized_language = _normalize_language(language)
    metadata = {
        "stage": capability.stage,
        "provider": capability.name,
        "criticality": capability.stage_criticality,
    }
    if normalized_language is not None:
        metadata["language"] = normalized_language

    if _language_supported(capability, normalized_language):
        return CapabilityMatch(
            capability=capability,
            language=normalized_language,
            should_run=True,
            reason="language_supported",
            metadata={**metadata, "reason": "language_supported"},
        )

    reason = (
        "language_disabled"
        if normalized_language in capability.disabled_languages
        else "language_unsupported"
    )
    if capability.stage_criticality == "required":
        raise ProviderCapabilityError(
            f"Required stage {capability.stage!r} provider {capability.name!r} "
            f"does not support language {normalized_language!r}"
        )
    return CapabilityMatch(
        capability=capability,
        language=normalized_language,
        should_run=False,
        reason=reason,
        metadata={**metadata, "reason": reason, "action": "skip"},
    )


def default_provider_capabilities() -> tuple[ProviderCapability, ...]:
    """Return all built-in static provider capability records."""

    return tuple(_DEFAULT_CAPABILITIES[key] for key in sorted(_DEFAULT_CAPABILITIES))
