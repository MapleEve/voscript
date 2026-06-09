"""Tests for the Python side of the Rust kernel bridge."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from providers.kernel_bridge import (
    RustKernelBridgeError,
    core_smoke,
    require_rust_core,
    rust_kernel_mode,
    rust_provider_paths_enabled,
)


def _fake_importer(module_name):
    assert module_name == "voscript_core"

    def _core_smoke(payload):
        return {
            "ok": True,
            "echoed": payload,
            "version": "0.8.0",
            "capabilities": {"core_smoke": True, "rust_extension": True},
        }

    return SimpleNamespace(core_smoke=_core_smoke)


def test_core_smoke_round_trips_safe_payload_through_imported_extension():
    payload = {"message": "hello", "items": [1, True, None]}

    result = core_smoke(payload, importer=_fake_importer)

    assert result["ok"] is True
    assert result["echoed"] == payload
    assert result["version"] == "0.8.0"
    assert result["capabilities"]["core_smoke"] is True


def test_missing_rust_extension_hard_fails():
    def _missing_importer(module_name):
        raise ModuleNotFoundError(module_name)

    with pytest.raises(RustKernelBridgeError, match="required but unavailable"):
        require_rust_core(importer=_missing_importer)


def test_core_smoke_call_failure_hard_fails():
    def _importer(module_name):
        assert module_name == "voscript_core"

        def _core_smoke(payload):
            raise RuntimeError("boom")

        return SimpleNamespace(core_smoke=_core_smoke)

    with pytest.raises(RustKernelBridgeError, match="core_smoke call failed"):
        core_smoke({"ok": True}, importer=_importer)


def test_core_smoke_invalid_response_hard_fails():
    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(core_smoke=lambda payload: {"ok": False})

    with pytest.raises(RustKernelBridgeError, match="missing keys"):
        core_smoke({}, importer=_importer)


def test_rust_kernel_mode_defaults_to_off_semantics():
    assert rust_kernel_mode("off") == "off"
    assert rust_provider_paths_enabled("off") is False
    assert rust_provider_paths_enabled("required") is True


def test_invalid_rust_kernel_mode_hard_fails():
    with pytest.raises(RustKernelBridgeError, match="Invalid RUST_KERNEL_MODE"):
        rust_kernel_mode("auto")
