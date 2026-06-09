"""Tests for the Python side of the Rust kernel bridge."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from providers.kernel_bridge import (
    RustKernelBridgeError,
    core_smoke,
    postprocess_segments,
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
            "version": "0.8.2",
            "capabilities": {"core_smoke": True, "rust_extension": True},
        }

    return SimpleNamespace(core_smoke=_core_smoke)


def test_core_smoke_round_trips_safe_payload_through_imported_extension():
    payload = {"message": "hello", "items": [1, True, None]}

    result = core_smoke(payload, importer=_fake_importer)

    assert result["ok"] is True
    assert result["echoed"] == payload
    assert result["version"] == "0.8.2"
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


def test_postprocess_segments_round_trips_valid_kernel_response():
    def _importer(module_name):
        assert module_name == "voscript_core"

        def _postprocess_segments(payload):
            assert payload["aligned_segments"][0]["speaker"] == "SPEAKER_00"
            return {
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "text": "hello",
                        "speaker_label": "SPEAKER_00",
                        "speaker_id": None,
                        "speaker_name": "SPEAKER_00",
                        "similarity": 0,
                        "words": [
                            {
                                "word": "hello",
                                "start": 0.0,
                                "end": 1.0,
                                "score": 0.0,
                            }
                        ],
                    }
                ],
                "unique_speakers": ["SPEAKER_00"],
            }

        return SimpleNamespace(postprocess_segments=_postprocess_segments)

    result = postprocess_segments(
        {
            "aligned_segments": [{"speaker": "SPEAKER_00"}],
            "speaker_map": {},
        },
        importer=_importer,
    )

    assert result["segments"][0]["id"] == 0
    assert result["segments"][0]["similarity"] == 0.0
    assert result["unique_speakers"] == ["SPEAKER_00"]


def test_postprocess_segments_invalid_response_hard_fails():
    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(postprocess_segments=lambda payload: {"segments": []})

    with pytest.raises(RustKernelBridgeError, match="missing keys"):
        postprocess_segments(
            {"aligned_segments": [], "speaker_map": {}},
            importer=_importer,
        )
