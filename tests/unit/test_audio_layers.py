"""Unit tests for the new audio provider / infra layering."""

from __future__ import annotations

import json
from pathlib import Path

import infra.audio.hash_index as hash_index_module
import infra.audio as audio_infra
import providers
from infra.audio import JsonAudioArtifactIndex
from pipeline.contracts import (
    AudioEnhancementRequest,
    AudioNormalizationRequest,
    UploadPersistenceRequest,
)


def test_audio_layer_entrypoints_point_to_new_modules():
    assert providers.convert_to_wav.__module__ == "providers.normalize"
    assert providers.maybe_denoise.__module__ == "providers.enhance"
    assert audio_infra.lookup_hash.__module__ == "infra.audio.hash_index"
    assert audio_infra.register_hash.__module__ == "infra.audio.hash_index"
    assert audio_infra.safe_tr_dir.__module__ == "infra.audio.paths"


def test_contract_entrypoints_are_instantiable():
    normalization = AudioNormalizationRequest(input_path=Path("sample.mp3"))
    enhancement = AudioEnhancementRequest(wav_path=Path("sample.wav"))
    persistence = UploadPersistenceRequest(
        file=None,
        save_path=Path("sample.wav"),
        max_bytes=1024,
        chunk_size=256,
    )

    assert normalization.target_format == "wav"
    assert enhancement.wav_path.name == "sample.wav"
    assert persistence.chunk_size == 256


def test_hash_index_infra_requires_completed_result(monkeypatch, tmp_path):
    monkeypatch.setattr(hash_index_module, "TRANSCRIPTIONS_DIR", tmp_path)

    store = JsonAudioArtifactIndex(index_path=tmp_path / "hash_index.json")
    store.register("hash-a", "tr_missing")
    assert store.lookup("hash-a") is None

    tr_dir = tmp_path / "tr_ready"
    tr_dir.mkdir(parents=True, exist_ok=True)
    (tr_dir / "result.json").write_text(json.dumps({"id": "tr_ready"}))

    store.register("hash-b", "tr_ready")
    assert store.lookup("hash-b") == "tr_ready"
