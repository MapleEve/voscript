"""Contract tests for artifact manifest, status, and schema helpers."""

from __future__ import annotations

import pytest

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
