"""Contract tests for artifact manifest, status, and schema helpers."""

from __future__ import annotations

import pytest

import pipeline.contracts as pipeline_contracts
from infra.job_status import build_status_payload, normalize_status_payload
from pipeline.contracts import (
    ARTIFACT_MANIFEST_VERSION,
    ArtifactManifestEntry,
    attach_optional_schema_version,
    build_artifact_manifest,
    empty_artifact_manifest,
    normalize_artifact_manifest,
    read_optional_schema_version,
)
from pipeline.registry import available_stage_slots


def test_artifact_manifest_builds_public_safe_known_categories_only():
    manifest = build_artifact_manifest(
        stable=[
            ArtifactManifestEntry(
                name="result",
                filename="result.json",
                role="primary_result",
                media_type="application/json",
                required_for_result=True,
            )
        ],
        optional=[
            ArtifactManifestEntry(
                name="speaker_embedding",
                filename="emb_SPEAKER_00.npy",
                role="speaker_embedding",
                media_type="application/octet-stream",
                speaker_label="SPEAKER_00",
            )
        ],
    )

    assert manifest == {
        "manifest_version": ARTIFACT_MANIFEST_VERSION,
        "stable": [
            {
                "name": "result",
                "filename": "result.json",
                "role": "primary_result",
                "media_type": "application/json",
                "required_for_result": True,
            }
        ],
        "optional": [
            {
                "name": "speaker_embedding",
                "filename": "emb_SPEAKER_00.npy",
                "role": "speaker_embedding",
                "media_type": "application/octet-stream",
                "required_for_result": False,
                "speaker_label": "SPEAKER_00",
            }
        ],
        "experimental": [],
    }


def test_artifact_manifest_rejects_generated_local_paths():
    with pytest.raises(ValueError, match="filename must not include"):
        build_artifact_manifest(
            stable=[
                ArtifactManifestEntry(
                    name="result",
                    filename="private/result.json",
                    role="primary_result",
                    media_type="application/json",
                )
            ]
        )


def test_normalize_artifact_manifest_tolerates_legacy_unknown_and_bad_entries():
    assert normalize_artifact_manifest(None) == empty_artifact_manifest()

    manifest = normalize_artifact_manifest(
        {
            "manifest_version": "future.v2",
            "stable": [
                {
                    "name": "result",
                    "filename": "result.json",
                    "role": "primary_result",
                    "media_type": "application/json",
                    "required_for_result": True,
                    "local_path": "private/result.json",
                    "speaker_id": "example-speaker-id",
                },
                {"name": "bad", "filename": "../secret", "role": "x"},
                "unknown-entry",
            ],
            "future_category": [{"path": "private/result.json"}],
        }
    )

    assert manifest["manifest_version"] == "future.v2"
    assert manifest["stable"] == [
        {
            "name": "result",
            "filename": "result.json",
            "role": "primary_result",
            "media_type": "application/json",
            "required_for_result": True,
        }
    ]
    assert manifest["optional"] == []
    assert manifest["experimental"] == []
    assert "local_path" not in manifest["stable"][0]
    assert "speaker_id" not in manifest["stable"][0]


def test_status_payload_build_and_legacy_normalization_keep_shape():
    payload = build_status_payload(
        "queued",
        filename="private/audio.wav",
        updated_at="2026-06-09T00:00:00+00:00",
    )

    assert payload == {
        "status": "queued",
        "updated_at": "2026-06-09T00:00:00+00:00",
        "error": None,
        "filename": "audio.wav",
    }

    legacy = normalize_status_payload(
        {
            "status": "transcribing",
            "updated_at": "2026-06-09T00:00:00+00:00",
            "filename": "C:\\private\\legacy.wav",
            "internal_path": "private/legacy.wav",
        }
    )
    assert legacy == {
        "status": "transcribing",
        "updated_at": "2026-06-09T00:00:00+00:00",
        "error": None,
        "filename": "legacy.wav",
    }
    assert normalize_status_payload({"status": "mystery"})["status"] == "failed"


def test_schema_version_is_optional_first_for_legacy_artifacts():
    assert read_optional_schema_version({"id": "tr_legacy"}) is None
    assert read_optional_schema_version({"schema_version": "result.v1"}) == "result.v1"
    assert attach_optional_schema_version({"id": "tr_legacy"}, None) == {
        "id": "tr_legacy"
    }
    assert attach_optional_schema_version({"id": "tr_new"}, "result.v1") == {
        "id": "tr_new",
        "schema_version": "result.v1",
    }
    with pytest.raises(ValueError, match="schema_version"):
        read_optional_schema_version({"schema_version": "../private"})


def test_pipeline_metadata_contract_covers_stable_stage_order_and_control_keys():
    assert hasattr(pipeline_contracts, "PIPELINE_METADATA_CONTRACT")
    assert hasattr(pipeline_contracts, "PIPELINE_METADATA_CONTROL_KEYS")
    assert hasattr(pipeline_contracts, "PIPELINE_METADATA_PATH_CONTRACT")
    assert hasattr(pipeline_contracts, "PIPELINE_METADATA_PUBLIC_PATHS")
    assert hasattr(pipeline_contracts, "PIPELINE_METADATA_STAGE_KEYS")

    PIPELINE_METADATA_CONTRACT = pipeline_contracts.PIPELINE_METADATA_CONTRACT
    PIPELINE_METADATA_CONTROL_KEYS = pipeline_contracts.PIPELINE_METADATA_CONTROL_KEYS
    PIPELINE_METADATA_PATH_CONTRACT = pipeline_contracts.PIPELINE_METADATA_PATH_CONTRACT
    PIPELINE_METADATA_PUBLIC_PATHS = pipeline_contracts.PIPELINE_METADATA_PUBLIC_PATHS
    PIPELINE_METADATA_STAGE_KEYS = pipeline_contracts.PIPELINE_METADATA_STAGE_KEYS

    assert PIPELINE_METADATA_STAGE_KEYS == available_stage_slots()
    assert PIPELINE_METADATA_CONTROL_KEYS == (
        "executed_stages",
        "selected_providers",
        "provider_capabilities",
        "stage_timings",
    )
    assert PIPELINE_METADATA_PUBLIC_PATHS == ("diarization.alignment",)

    for key in (*PIPELINE_METADATA_CONTROL_KEYS, *PIPELINE_METADATA_STAGE_KEYS):
        entry = PIPELINE_METADATA_CONTRACT[key]
        assert entry.owner
        assert entry.writers
        assert isinstance(entry.public, bool)
        assert isinstance(entry.allow_overwrite, bool)

    alignment_entry = PIPELINE_METADATA_PATH_CONTRACT["diarization.alignment"]
    assert alignment_entry.owner == "diarization"
    assert alignment_entry.public is True
    assert alignment_entry.allow_overwrite is False


def test_public_alignment_metadata_normalizer_keeps_safe_scalars_only(tmp_path):
    assert hasattr(pipeline_contracts, "normalize_public_alignment_metadata")
    normalize_public_alignment_metadata = (
        pipeline_contracts.normalize_public_alignment_metadata
    )

    normalized = normalize_public_alignment_metadata(
        {
            "status": "skipped",
            "reason": "duration_budget_exceeded",
            "model": "org/model",
            "duration_s": 12.5,
            "max_duration_s": 60,
            "cache_only": False,
            "device": "cpu",
            "language": "zh",
            "model_path": str(tmp_path / "private-model"),
            "exception": RuntimeError("hidden"),
            "debug": {"path": str(tmp_path)},
            "segments": ["not public"],
        }
    )

    assert normalized == {
        "status": "skipped",
        "reason": "duration_budget_exceeded",
        "model": "org/model",
        "duration_s": 12.5,
        "max_duration_s": 60,
        "cache_only": False,
        "device": "cpu",
    }
