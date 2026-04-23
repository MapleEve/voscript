"""Unified lazy registry for stable pipeline stages and pluggable providers."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Callable

from .contracts import ProviderNotFoundError, StageNotFoundError

_STEP_ALIASES = {
    "input_normalization": "normalize",
    "enhancement": "enhance",
}

_DEFAULT_STAGE_IMPORTS = {
    "ingest": "pipeline.stages.ingest:run",
    "normalize": "pipeline.stages.normalize:run",
    "enhance": "pipeline.stages.enhance:run",
    "vad": "pipeline.stages.vad:run",
    "asr": "pipeline.stages.asr:run",
    "diarization": "pipeline.stages.diarization:run",
    "embedding": "pipeline.stages.embedding:run",
    "voiceprint_match": "pipeline.stages.voiceprint_match:run",
    "punc": "pipeline.stages.punc:run",
    "postprocess": "pipeline.stages.postprocess:run",
    "artifacts": "pipeline.stages.artifacts:run",
}

_DEFAULT_PROVIDER_IMPORTS = {
    "ingest": {
        "default": "providers.ingest.default:default_ingest_provider",
    },
    "normalize": {
        "default": "providers.normalize.default:default_normalize_provider",
    },
    "enhance": {
        "default": "providers.enhance.default:default_enhance_provider",
    },
    "vad": {
        "default": "providers.vad.default:default_vad_provider",
    },
    "asr": {
        "default": "providers.asr.default:default_asr_provider",
    },
    "diarization": {
        "default": "providers.diarization.default:default_diarization_provider",
    },
    "embedding": {
        "default": "providers.embedding.default:default_speaker_embedding_provider",
    },
    "voiceprint_match": {
        "default": "providers.voiceprint_match.default:default_voiceprint_match_provider",
    },
    "punc": {
        "default": "providers.punc.default:default_punc_provider",
    },
    "postprocess": {
        "default": "providers.postprocess.default:default_postprocess_provider",
    },
    "artifacts": {
        "default": "providers.artifacts.default:default_artifacts_provider",
    },
}

_PROVIDER_OVERRIDES: dict[str, dict[str, Any]] = {}


def _normalize_token(value: str, *, field_name: str) -> str:
    token = value.strip().lower().replace("-", "_")
    if not token:
        raise ValueError(f"{field_name} must not be empty")
    return token


def canonical_step_name(step: str) -> str:
    """Map compatibility aliases onto the canonical stable step names."""

    token = _normalize_token(step, field_name="step")
    return _STEP_ALIASES.get(token, token)


def _load_object(import_path: str) -> Any:
    module_name, _, attr_name = import_path.partition(":")
    if not module_name or not attr_name:
        raise ValueError(f"Invalid provider import path: {import_path!r}")
    module = import_module(module_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as exc:
        raise ProviderNotFoundError(
            f"Provider import target {import_path!r} was not found"
        ) from exc


def register_provider(step: str, name: str, provider: Any) -> None:
    """Register or override a provider implementation for a pipeline step."""

    step_key = canonical_step_name(step)
    name_key = _normalize_token(name, field_name="name")
    _PROVIDER_OVERRIDES.setdefault(step_key, {})[name_key] = provider


def unregister_provider(step: str, name: str) -> None:
    """Remove a test or runtime override provider."""

    step_key = canonical_step_name(step)
    name_key = _normalize_token(name, field_name="name")
    step_overrides = _PROVIDER_OVERRIDES.get(step_key)
    if not step_overrides:
        return
    step_overrides.pop(name_key, None)
    if not step_overrides:
        _PROVIDER_OVERRIDES.pop(step_key, None)


def resolve_provider(step: str, name: str = "default") -> Any:
    """Resolve a provider object by stable step name and implementation name."""

    step_key = canonical_step_name(step)
    name_key = _normalize_token(name, field_name="name")

    override = _PROVIDER_OVERRIDES.get(step_key, {}).get(name_key)
    if override is not None:
        return override

    import_path = _DEFAULT_PROVIDER_IMPORTS.get(step_key, {}).get(name_key)
    if import_path is None:
        raise ProviderNotFoundError(
            f"No provider registered for step={step_key!r} name={name_key!r}"
        )
    return _load_object(import_path)


def available_providers(step: str) -> tuple[str, ...]:
    """List available provider names for a stable pipeline step."""

    step_key = canonical_step_name(step)
    names = set(_DEFAULT_PROVIDER_IMPORTS.get(step_key, {}))
    names.update(_PROVIDER_OVERRIDES.get(step_key, {}))
    return tuple(sorted(names))


def available_stage_slots() -> tuple[str, ...]:
    """List the stable pipeline slots in execution order."""

    return tuple(_DEFAULT_STAGE_IMPORTS)


def resolve_stage(slot: str) -> Callable[[Any], None]:
    """Resolve the callable backing a stable stage slot."""

    slot_name = canonical_step_name(slot)
    import_path = _DEFAULT_STAGE_IMPORTS.get(slot_name)
    if import_path is None:
        raise StageNotFoundError(f"Unknown stage slot: {slot_name!r}")
    return _load_object(import_path)


__all__ = [
    "ProviderNotFoundError",
    "StageNotFoundError",
    "available_providers",
    "available_stage_slots",
    "canonical_step_name",
    "register_provider",
    "resolve_provider",
    "resolve_stage",
    "unregister_provider",
]
