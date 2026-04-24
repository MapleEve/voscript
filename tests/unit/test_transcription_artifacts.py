"""Failure-path tests for persisted transcription artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import infra.transcription_artifacts as transcription_artifacts


def test_persist_transcription_artifacts_propagates_result_write_failure(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "tr_result_fail"

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(transcription_artifacts, "_atomic_write_json", fail_write)

    with pytest.raises(OSError, match="disk full"):
        transcription_artifacts.persist_transcription_artifacts(
            output_dir,
            {"id": "tr_result_fail", "segments": []},
            {"SPEAKER_00": np.array([0.1, 0.2], dtype=np.float32)},
        )

    assert not (output_dir / "result.json").exists()
    assert list(output_dir.glob("emb_*.npy")) == []


def test_persist_transcription_artifacts_cleans_partial_embeddings_on_failure(
    tmp_path, monkeypatch
):
    output_dir = tmp_path / "tr_embedding_fail"
    real_save = transcription_artifacts.np.save
    save_calls: list[Path] = []

    def flaky_save(path, array):
        save_path = Path(path)
        save_calls.append(save_path)
        if len(save_calls) == 1:
            real_save(save_path, array)
            return
        save_path.write_bytes(b"partial")
        raise OSError("embedding write failed")

    monkeypatch.setattr(transcription_artifacts.np, "save", flaky_save)

    with pytest.raises(OSError, match="embedding write failed"):
        transcription_artifacts.persist_transcription_artifacts(
            output_dir,
            {"id": "tr_embedding_fail", "segments": []},
            {
                "SPEAKER_00": np.array([0.1, 0.2], dtype=np.float32),
                "SPEAKER_01": np.array([0.3, 0.4], dtype=np.float32),
            },
        )

    assert len(save_calls) == 2
    assert not (output_dir / "result.json").exists()
    assert list(output_dir.glob("emb_*.npy")) == []
