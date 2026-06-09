"""Fail-closed bridge from Python providers to native Rust kernels."""

from __future__ import annotations

from collections.abc import Mapping
from importlib import import_module
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
