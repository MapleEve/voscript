"""Fail-closed bridge from Python providers to native Rust kernels."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
from math import isfinite
from types import ModuleType
from typing import Any, Callable

from config import RUST_KERNEL_MODE

RUST_KERNEL_MODE_OFF = "off"
RUST_KERNEL_MODE_REQUIRED = "required"
_VALID_RUST_KERNEL_MODES = {RUST_KERNEL_MODE_OFF, RUST_KERNEL_MODE_REQUIRED}


class RustKernelBridgeError(RuntimeError):
    """Raised when a selected Rust-backed kernel path cannot run safely."""


def rust_kernel_mode(value: str | None = None) -> str:
    """Return the normalized Rust kernel mode or fail on invalid config."""

    raw = RUST_KERNEL_MODE if value is None else value
    mode = raw.strip().lower()
    if mode not in _VALID_RUST_KERNEL_MODES:
        valid = ", ".join(sorted(_VALID_RUST_KERNEL_MODES))
        raise RustKernelBridgeError(
            f"Invalid RUST_KERNEL_MODE={raw!r}; expected one of: {valid}"
        )
    return mode


def rust_provider_paths_enabled(value: str | None = None) -> bool:
    """Whether provider paths may select Rust-backed kernels."""

    return rust_kernel_mode(value) == RUST_KERNEL_MODE_REQUIRED


def require_rust_core(
    importer: Callable[[str], ModuleType] = import_module,
) -> ModuleType:
    """Import the native extension, mapping all failures to a hard error."""

    try:
        return importer("voscript_core")
    except Exception as exc:
        raise RustKernelBridgeError(
            "Rust kernel extension is required but unavailable"
        ) from exc


def _validate_smoke_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise RustKernelBridgeError("Rust core_smoke returned a non-mapping response")

    result = dict(response)
    required_keys = {"ok", "echoed", "version", "capabilities"}
    missing = sorted(required_keys.difference(result))
    if missing:
        raise RustKernelBridgeError(
            f"Rust core_smoke response missing keys: {', '.join(missing)}"
        )
    if result["ok"] is not True:
        raise RustKernelBridgeError("Rust core_smoke did not report ok=true")
    if not isinstance(result["capabilities"], Mapping):
        raise RustKernelBridgeError("Rust core_smoke capabilities must be a mapping")
    return result


def _validate_voiceprint_score_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise RustKernelBridgeError(
            "Rust voiceprint_score returned a non-mapping response"
        )

    result = dict(response)
    required_keys = {
        "matched_id",
        "matched_name",
        "similarity",
        "reason",
        "asnorm_active",
        "asnorm_reason",
        "candidates",
    }
    missing = sorted(required_keys.difference(result))
    if missing:
        raise RustKernelBridgeError(
            f"Rust voiceprint_score response missing keys: {', '.join(missing)}"
        )
    if not isinstance(result["reason"], str) or not result["reason"]:
        raise RustKernelBridgeError("Rust voiceprint_score reason must be non-empty")
    if not isinstance(result["asnorm_active"], bool):
        raise RustKernelBridgeError("Rust voiceprint_score asnorm_active must be bool")
    if not isinstance(result["asnorm_reason"], str) or not result["asnorm_reason"]:
        raise RustKernelBridgeError(
            "Rust voiceprint_score asnorm_reason must be non-empty"
        )
    if not isinstance(result["candidates"], list):
        raise RustKernelBridgeError("Rust voiceprint_score candidates must be a list")
    result["candidates"] = [
        _validate_voiceprint_score_candidate_response(candidate)
        for candidate in result["candidates"]
    ]
    try:
        result["similarity"] = float(result["similarity"])
    except (TypeError, ValueError) as exc:
        raise RustKernelBridgeError(
            "Rust voiceprint_score similarity must be numeric"
        ) from exc
    if not isfinite(result["similarity"]):
        raise RustKernelBridgeError("Rust voiceprint_score similarity must be finite")
    return result


def _validate_voiceprint_score_candidate_response(candidate: Any) -> dict[str, Any]:
    if not isinstance(candidate, Mapping):
        raise RustKernelBridgeError(
            "Rust voiceprint_score candidate returned a non-mapping response"
        )

    result = dict(candidate)
    required_keys = {
        "speaker_id",
        "name",
        "raw_similarity",
        "similarity",
        "effective_threshold",
        "score_method",
        "sample_count",
        "sample_spread",
    }
    missing = sorted(required_keys.difference(result))
    if missing:
        raise RustKernelBridgeError(
            "Rust voiceprint_score candidate response missing keys: "
            + ", ".join(missing)
        )
    for key in ("speaker_id", "name", "score_method"):
        if not isinstance(result[key], str) or not result[key]:
            raise RustKernelBridgeError(
                f"Rust voiceprint_score candidate {key} must be non-empty"
            )
    for key in ("raw_similarity", "similarity", "effective_threshold"):
        try:
            result[key] = float(result[key])
        except (TypeError, ValueError) as exc:
            raise RustKernelBridgeError(
                f"Rust voiceprint_score candidate {key} must be numeric"
            ) from exc
        if not isfinite(result[key]):
            raise RustKernelBridgeError(
                f"Rust voiceprint_score candidate {key} must be finite"
            )
    try:
        result["sample_count"] = int(result["sample_count"])
    except (TypeError, ValueError) as exc:
        raise RustKernelBridgeError(
            "Rust voiceprint_score candidate sample_count must be integer-like"
        ) from exc
    if result["sample_count"] < 0:
        raise RustKernelBridgeError(
            "Rust voiceprint_score candidate sample_count must be non-negative"
        )
    if result["sample_spread"] is not None:
        try:
            result["sample_spread"] = float(result["sample_spread"])
        except (TypeError, ValueError) as exc:
            raise RustKernelBridgeError(
                "Rust voiceprint_score candidate sample_spread must be numeric"
            ) from exc
        if not isfinite(result["sample_spread"]):
            raise RustKernelBridgeError(
                "Rust voiceprint_score candidate sample_spread must be finite"
            )
    return result


def core_smoke(
    payload: Any,
    importer: Callable[[str], ModuleType] = import_module,
) -> dict[str, Any]:
    """Call the native extension smoke function and fail closed on errors."""

    rust_core = require_rust_core(importer=importer)
    try:
        response = rust_core.core_smoke(payload)
    except Exception as exc:
        raise RustKernelBridgeError("Rust core_smoke call failed") from exc
    return _validate_smoke_response(response)


def voiceprint_score(
    payload: dict[str, Any],
    importer: Callable[[str], ModuleType] = import_module,
) -> dict[str, Any]:
    """Call the native voiceprint scoring kernel and fail closed on errors."""

    rust_core = require_rust_core(importer=importer)
    try:
        response = rust_core.voiceprint_score(payload)
    except Exception as exc:
        raise RustKernelBridgeError("Rust voiceprint_score call failed") from exc
    return _validate_voiceprint_score_response(response)
