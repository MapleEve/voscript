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


def _validate_postprocess_segments_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise RustKernelBridgeError(
            "Rust postprocess_segments returned a non-mapping response"
        )

    result = dict(response)
    required_keys = {"segments", "unique_speakers"}
    missing = sorted(required_keys.difference(result))
    if missing:
        raise RustKernelBridgeError(
            f"Rust postprocess_segments response missing keys: {', '.join(missing)}"
        )
    if not isinstance(result["segments"], list):
        raise RustKernelBridgeError("Rust postprocess_segments segments must be a list")
    if not isinstance(result["unique_speakers"], list) or not all(
        isinstance(speaker, str) and speaker for speaker in result["unique_speakers"]
    ):
        raise RustKernelBridgeError(
            "Rust postprocess_segments unique_speakers must be non-empty strings"
        )
    result["segments"] = [
        _validate_postprocess_segment_response(segment)
        for segment in result["segments"]
    ]
    return result


def _validate_artifact_manifest_contract_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise RustKernelBridgeError(
            "Rust artifact_manifest_contract returned a non-mapping response"
        )

    result = dict(response)
    required_keys = {"manifest_version", "stable", "optional", "experimental"}
    missing = sorted(required_keys.difference(result))
    if missing:
        raise RustKernelBridgeError(
            "Rust artifact_manifest_contract response missing keys: "
            + ", ".join(missing)
        )
    if (
        not isinstance(result["manifest_version"], str)
        or not result["manifest_version"]
    ):
        raise RustKernelBridgeError(
            "Rust artifact_manifest_contract manifest_version must be non-empty"
        )
    for category in ("stable", "optional", "experimental"):
        if not isinstance(result[category], list):
            raise RustKernelBridgeError(
                f"Rust artifact_manifest_contract {category} must be a list"
            )
        result[category] = [
            _validate_artifact_manifest_entry_response(entry)
            for entry in result[category]
        ]
    return result


def _validate_artifact_manifest_entry_response(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        raise RustKernelBridgeError(
            "Rust artifact_manifest_contract entry returned a non-mapping response"
        )
    result = dict(entry)
    required_keys = {
        "name",
        "filename",
        "role",
        "media_type",
        "required_for_result",
    }
    missing = sorted(required_keys.difference(result))
    if missing:
        raise RustKernelBridgeError(
            "Rust artifact_manifest_contract entry missing keys: " + ", ".join(missing)
        )
    for key in ("name", "filename", "role", "media_type"):
        if not isinstance(result[key], str) or not result[key]:
            raise RustKernelBridgeError(
                f"Rust artifact_manifest_contract entry {key} must be non-empty"
            )
    if (
        "/" in result["filename"]
        or "\\" in result["filename"]
        or "://" in result["filename"]
    ):
        raise RustKernelBridgeError(
            "Rust artifact_manifest_contract filename must not expose a path"
        )
    if not isinstance(result["required_for_result"], bool):
        raise RustKernelBridgeError(
            "Rust artifact_manifest_contract required_for_result must be bool"
        )
    if "speaker_label" in result and not isinstance(result["speaker_label"], str):
        raise RustKernelBridgeError(
            "Rust artifact_manifest_contract speaker_label must be string"
        )
    return result


def _validate_status_payload_contract_response(response: Any) -> dict[str, Any]:
    if not isinstance(response, Mapping):
        raise RustKernelBridgeError(
            "Rust status_payload_contract returned a non-mapping response"
        )

    result = dict(response)
    required_keys = {"status", "updated_at", "error"}
    missing = sorted(required_keys.difference(result))
    if missing:
        raise RustKernelBridgeError(
            "Rust status_payload_contract response missing keys: " + ", ".join(missing)
        )
    if not isinstance(result["status"], str) or not result["status"]:
        raise RustKernelBridgeError("Rust status_payload_contract status is invalid")
    if not isinstance(result["updated_at"], str) or not result["updated_at"]:
        raise RustKernelBridgeError(
            "Rust status_payload_contract updated_at is invalid"
        )
    if result["error"] is not None and not isinstance(result["error"], str):
        raise RustKernelBridgeError(
            "Rust status_payload_contract error must be string or null"
        )
    if "filename" in result and not isinstance(result["filename"], str):
        raise RustKernelBridgeError(
            "Rust status_payload_contract filename must be string"
        )
    return result


def _validate_postprocess_segment_response(segment: Any) -> dict[str, Any]:
    if not isinstance(segment, Mapping):
        raise RustKernelBridgeError(
            "Rust postprocess_segments segment returned a non-mapping response"
        )

    result = dict(segment)
    required_keys = {
        "id",
        "start",
        "end",
        "text",
        "speaker_label",
        "speaker_id",
        "speaker_name",
        "similarity",
    }
    missing = sorted(required_keys.difference(result))
    if missing:
        raise RustKernelBridgeError(
            "Rust postprocess_segments segment missing keys: " + ", ".join(missing)
        )
    try:
        result["id"] = int(result["id"])
    except (TypeError, ValueError) as exc:
        raise RustKernelBridgeError(
            "Rust postprocess_segments segment id must be integer-like"
        ) from exc
    if result["id"] < 0:
        raise RustKernelBridgeError(
            "Rust postprocess_segments segment id must be non-negative"
        )
    for key in ("start", "end", "similarity"):
        try:
            result[key] = float(result[key])
        except (TypeError, ValueError) as exc:
            raise RustKernelBridgeError(
                f"Rust postprocess_segments segment {key} must be numeric"
            ) from exc
        if not isfinite(result[key]):
            raise RustKernelBridgeError(
                f"Rust postprocess_segments segment {key} must be finite"
            )
    for key in ("text", "speaker_label", "speaker_name"):
        if not isinstance(result[key], str):
            raise RustKernelBridgeError(
                f"Rust postprocess_segments segment {key} must be a string"
            )
    if not result["speaker_label"] or not result["speaker_name"]:
        raise RustKernelBridgeError(
            "Rust postprocess_segments segment speaker labels must be non-empty"
        )
    if result["speaker_id"] is not None and not isinstance(result["speaker_id"], str):
        raise RustKernelBridgeError(
            "Rust postprocess_segments segment speaker_id must be a string or null"
        )
    if "words" in result:
        if not isinstance(result["words"], list):
            raise RustKernelBridgeError(
                "Rust postprocess_segments segment words must be a list"
            )
        result["words"] = [
            _validate_postprocess_word_response(word) for word in result["words"]
        ]
    return result


def _validate_postprocess_word_response(word: Any) -> dict[str, Any]:
    if not isinstance(word, Mapping):
        raise RustKernelBridgeError(
            "Rust postprocess_segments word returned a non-mapping response"
        )
    result = dict(word)
    required_keys = {"word", "start", "end", "score"}
    missing = sorted(required_keys.difference(result))
    if missing:
        raise RustKernelBridgeError(
            "Rust postprocess_segments word missing keys: " + ", ".join(missing)
        )
    if not isinstance(result["word"], str):
        raise RustKernelBridgeError(
            "Rust postprocess_segments word text must be string"
        )
    for key in ("start", "end", "score"):
        try:
            result[key] = float(result[key])
        except (TypeError, ValueError) as exc:
            raise RustKernelBridgeError(
                f"Rust postprocess_segments word {key} must be numeric"
            ) from exc
        if not isfinite(result[key]):
            raise RustKernelBridgeError(
                f"Rust postprocess_segments word {key} must be finite"
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


def artifact_manifest_contract(
    payload: dict[str, Any],
    importer: Callable[[str], ModuleType] = import_module,
) -> dict[str, Any]:
    """Call the native artifact manifest helper contract."""

    rust_core = require_rust_core(importer=importer)
    try:
        response = rust_core.artifact_manifest_contract(payload)
    except Exception as exc:
        raise RustKernelBridgeError(
            "Rust artifact_manifest_contract call failed"
        ) from exc
    return _validate_artifact_manifest_contract_response(response)


def status_payload_contract(
    payload: dict[str, Any],
    importer: Callable[[str], ModuleType] = import_module,
) -> dict[str, Any]:
    """Call the native status payload helper contract."""

    rust_core = require_rust_core(importer=importer)
    try:
        response = rust_core.status_payload_contract(payload)
    except Exception as exc:
        raise RustKernelBridgeError("Rust status_payload_contract call failed") from exc
    return _validate_status_payload_contract_response(response)


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


def postprocess_segments(
    payload: dict[str, Any],
    importer: Callable[[str], ModuleType] = import_module,
) -> dict[str, Any]:
    """Call the native result-segment post-processing kernel."""

    rust_core = require_rust_core(importer=importer)
    try:
        response = rust_core.postprocess_segments(payload)
    except Exception as exc:
        raise RustKernelBridgeError("Rust postprocess_segments call failed") from exc
    return _validate_postprocess_segments_response(response)
