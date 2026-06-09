"""Tests for the Python side of the Rust kernel bridge."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from providers.kernel_bridge import (
    RustKernelBridgeError,
    artifact_manifest_contract,
    core_smoke,
    postprocess_segments,
    require_rust_core,
    rust_kernel_mode,
    rust_provider_paths_enabled,
    status_payload_contract,
)


def _fake_importer(module_name):
    assert module_name == "voscript_core"

    def _core_smoke(payload):
        return {
            "ok": True,
            "echoed": payload,
            "version": "0.8.3",
            "capabilities": {"core_smoke": True, "rust_extension": True},
        }

    return SimpleNamespace(core_smoke=_core_smoke)


def test_core_smoke_round_trips_safe_payload_through_imported_extension():
    payload = {"message": "hello", "items": [1, True, None]}

    result = core_smoke(payload, importer=_fake_importer)

    assert result["ok"] is True
    assert result["echoed"] == payload
    assert result["version"] == "0.8.3"
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


def test_artifact_manifest_contract_round_trips_valid_kernel_response():
    def _importer(module_name):
        assert module_name == "voscript_core"

        def _artifact_manifest_contract(payload):
            assert payload["stable"][0]["filename"] == "result.json"
            return {
                "manifest_version": "artifact_manifest.v1",
                "stable": [
                    {
                        "name": "result",
                        "filename": "result.json",
                        "role": "primary_result",
                        "media_type": "application/json",
                        "required_for_result": True,
                    }
                ],
                "optional": [],
                "experimental": [],
            }

        return SimpleNamespace(artifact_manifest_contract=_artifact_manifest_contract)

    result = artifact_manifest_contract(
        {
            "manifest_version": "artifact_manifest.v1",
            "stable": [
                {
                    "name": "result",
                    "filename": "result.json",
                    "role": "primary_result",
                    "media_type": "application/json",
                    "required_for_result": True,
                }
            ],
            "optional": [],
            "experimental": [],
        },
        importer=_importer,
    )

    assert result["stable"][0]["required_for_result"] is True


def test_artifact_manifest_contract_rejects_path_leak_response():
    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(
            artifact_manifest_contract=lambda payload: {
                "manifest_version": "artifact_manifest.v1",
                "stable": [
                    {
                        "name": "result",
                        "filename": "private/result.json",
                        "role": "primary_result",
                        "media_type": "application/json",
                        "required_for_result": True,
                    }
                ],
                "optional": [],
                "experimental": [],
            }
        )

    with pytest.raises(RustKernelBridgeError, match="filename must not expose"):
        artifact_manifest_contract({"stable": []}, importer=_importer)


def test_status_payload_contract_round_trips_valid_kernel_response():
    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(
            status_payload_contract=lambda payload: {
                "status": "queued",
                "updated_at": "2026-06-09T00:00:00+00:00",
                "error": None,
                "filename": "audio.wav",
            }
        )

    result = status_payload_contract(
        {
            "status": "queued",
            "updated_at": "2026-06-09T00:00:00+00:00",
            "filename": "audio.wav",
        },
        importer=_importer,
    )

    assert result["status"] == "queued"
    assert result["error"] is None
    assert result["filename"] == "audio.wav"
