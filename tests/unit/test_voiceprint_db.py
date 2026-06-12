"""Voiceprint / dedup regressions that must run in CI.

Covers:
- debounce + rebuild-lock guardrails on VoiceprintDB
- auto rebuild persisting ``asnorm_cohort.npy`` for restart-time loading
- lifespan startup preferring a saved cohort over rebuilding
- true concurrent upload dedup (two simultaneous requests, one worker)
"""

from __future__ import annotations

import importlib
import io
import json
import sys
import threading
import time
import wave
import base64
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

_APP_DIR = Path(__file__).resolve().parents[2] / "app"


def _fresh_voiceprint_module():
    for name in list(sys.modules):
        if name == "voiceprints" or name.startswith("voiceprints."):
            sys.modules.pop(name, None)
    return importlib.import_module("voiceprints.db")


def _fresh_db(db_dir: Path):
    mod = _fresh_voiceprint_module()
    db_dir.mkdir(parents=True, exist_ok=True)
    return mod.VoiceprintDB(str(db_dir)), mod


def _unit_vec(seed: int, dim: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v


def _write_transcription_embedding(
    transcriptions_dir: Path,
    tr_id: str,
    emb: np.ndarray,
    label: str = "SPEAKER_00",
) -> Path:
    tr_dir = transcriptions_dir / tr_id
    tr_dir.mkdir(parents=True, exist_ok=True)
    (tr_dir / "result.json").write_text(
        json.dumps(
            {
                "id": tr_id,
                "speaker_embeddings": {label: emb.astype(np.float32).tolist()},
            }
        )
    )
    return tr_dir


def _wav_bytes(seconds: float = 0.2, sample_rate: int = 16000) -> bytes:
    frames = int(sample_rate * seconds)
    payload = io.BytesIO()
    with wave.open(payload, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * frames)
    return payload.getvalue()


def _fresh_main(monkeypatch, data_dir: Path):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    monkeypatch.chdir(_APP_DIR)

    for name in list(sys.modules):
        if name in {"main", "config"}:
            sys.modules.pop(name, None)
        elif name == "voiceprints" or name.startswith("voiceprints."):
            sys.modules.pop(name, None)
        elif name == "api" or name.startswith("api."):
            sys.modules.pop(name, None)
        elif name.startswith("services."):
            sys.modules.pop(name, None)

    voiceprint_mod = importlib.import_module("voiceprints.db")
    return voiceprint_mod


def test_maybe_rebuild_cohort_debounce_blocks(tmp_path):
    """maybe_rebuild_cohort must not rebuild before the debounce window expires."""
    db, _mod = _fresh_db(tmp_path / "voiceprints")
    db._cohort_generation = 1
    db._cohort_built_gen = 0
    db._cohort_last_enroll = time.monotonic()

    result = db.maybe_rebuild_cohort(str(tmp_path / "transcriptions"), debounce_s=30.0)
    assert result is False, "Expected debounce to block rebuild, got True"


def test_build_cohort_nonblocking_when_lock_held(tmp_path):
    """A concurrent caller must skip cohort rebuild instead of blocking."""
    db, _mod = _fresh_db(tmp_path / "voiceprints")
    acquired = db._cohort_rebuild_lock.acquire(blocking=False)
    assert acquired, "Expected rebuild lock to be acquirable while idle"

    try:
        result = db.build_cohort_from_transcriptions(str(tmp_path / "transcriptions"))
        assert isinstance(result, int), f"Expected int return, got {type(result)}"
    finally:
        db._cohort_rebuild_lock.release()


def test_auto_rebuild_persists_cohort_for_restart_load(tmp_path):
    """Auto rebuild must write asnorm_cohort.npy so a restart can load it directly."""
    transcriptions_dir = tmp_path / "transcriptions"
    voiceprints_dir = tmp_path / "voiceprints"
    emb = _unit_vec(11)
    _write_transcription_embedding(transcriptions_dir, "tr_auto_rebuild", emb)

    db, _mod = _fresh_db(voiceprints_dir)
    db._cohort_generation = 1
    db._cohort_built_gen = 0
    db._cohort_last_enroll = 0.0

    rebuilt = db.maybe_rebuild_cohort(str(transcriptions_dir), debounce_s=0.0)
    cohort_path = transcriptions_dir / "asnorm_cohort.npy"

    assert rebuilt is True, "Dirty cohort should rebuild once debounce has elapsed"
    assert cohort_path.exists(), (
        "Auto rebuild must persist transcriptions/asnorm_cohort.npy so startup can "
        "load the cohort after a process restart"
    )

    restarted_db, _mod = _fresh_db(voiceprints_dir)
    restarted_db.load_cohort(str(cohort_path))
    assert restarted_db.cohort_size == 1


def test_direct_load_marks_cohort_clean_and_auto_tick_does_not_shrink(tmp_path):
    """Loading a saved AS-norm cohort must not make the auto worker rebuild it."""
    transcriptions_dir = tmp_path / "transcriptions"
    cohort_path = transcriptions_dir / "asnorm_cohort.npy"
    saved = np.stack([_unit_vec(1000 + i) for i in range(65)]).astype(np.float32)
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cohort_path, saved)
    _write_transcription_embedding(transcriptions_dir, "tr_single", _unit_vec(2000))

    db, _mod = _fresh_db(tmp_path / "voiceprints")
    db.load_cohort(str(cohort_path))
    db._cohort_last_enroll = 0.0

    rebuilt = db.maybe_rebuild_cohort(str(transcriptions_dir), debounce_s=0.0)

    assert rebuilt is False, "A direct-loaded clean cohort should not auto-rebuild"
    assert db.cohort_size == 65
    assert np.load(cohort_path, allow_pickle=False).shape == (65, 256)


def test_auto_rebuild_keeps_larger_cohort_when_sources_are_too_small(tmp_path):
    """Dirty auto rebuild must not shrink a larger persisted AS-norm cohort."""
    transcriptions_dir = tmp_path / "transcriptions"
    cohort_path = transcriptions_dir / "asnorm_cohort.npy"
    saved = np.stack([_unit_vec(3000 + i) for i in range(65)]).astype(np.float32)
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cohort_path, saved)
    _write_transcription_embedding(transcriptions_dir, "tr_single", _unit_vec(4000))

    db, _mod = _fresh_db(tmp_path / "voiceprints")
    db.load_cohort(str(cohort_path))
    db._cohort_generation += 1
    db._cohort_last_enroll = 0.0

    rebuilt = db.maybe_rebuild_cohort(str(transcriptions_dir), debounce_s=0.0)

    assert rebuilt is False, "Auto rebuild should skip smaller source cohorts"
    assert db.cohort_size == 65
    assert np.load(cohort_path, allow_pickle=False).shape == (65, 256)


def test_auto_rebuild_keeps_larger_cohort_when_sources_are_empty(tmp_path):
    """Dirty auto rebuild must treat transcription cleanup as a no-shrink no-op."""
    transcriptions_dir = tmp_path / "transcriptions"
    cohort_path = transcriptions_dir / "asnorm_cohort.npy"
    saved = np.stack([_unit_vec(4500 + i) for i in range(65)]).astype(np.float32)
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cohort_path, saved)

    db, _mod = _fresh_db(tmp_path / "voiceprints")
    db.load_cohort(str(cohort_path))
    db._cohort_generation += 1
    db._cohort_last_enroll = 0.0

    rebuilt = db.maybe_rebuild_cohort(str(transcriptions_dir), debounce_s=0.0)

    assert rebuilt is False, "Auto rebuild should skip an empty source cohort"
    assert db.cohort_size == 65
    assert db._cohort_built_gen == db._cohort_generation
    assert np.load(cohort_path, allow_pickle=False).shape == (65, 256)


def test_manual_rebuild_can_replace_existing_cohort_with_available_sources(tmp_path):
    """The explicit rebuild API path keeps its existing force-like semantics."""
    transcriptions_dir = tmp_path / "transcriptions"
    cohort_path = transcriptions_dir / "asnorm_cohort.npy"
    saved = np.stack([_unit_vec(5000 + i) for i in range(65)]).astype(np.float32)
    cohort_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cohort_path, saved)
    _write_transcription_embedding(transcriptions_dir, "tr_single", _unit_vec(6000))

    db, _mod = _fresh_db(tmp_path / "voiceprints")
    db.load_cohort(str(cohort_path))

    rebuilt_size = db.build_cohort_from_transcriptions(str(transcriptions_dir))

    assert rebuilt_size == 1
    assert db.cohort_size == 1
    assert np.load(cohort_path, allow_pickle=False).shape == (1, 256)


def test_legacy_npy_voiceprint_store_migrates_to_sqlite(tmp_path):
    """A pre-SQLite voiceprint store must migrate avg and sample embeddings once."""
    db_dir = tmp_path / "voiceprints"
    db_dir.mkdir(parents=True)
    speaker_id = "spk_legacy"
    avg = _unit_vec(7000)
    samples = np.stack([avg, _unit_vec(7001)]).astype(np.float32)

    (db_dir / "index.json").write_text(
        json.dumps(
            {
                "speakers": {
                    speaker_id: {
                        "name": "Legacy",
                        "sample_count": 2,
                        "created_at": "2026-04-01T00:00:00",
                        "updated_at": "2026-04-02T00:00:00",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    np.save(db_dir / f"{speaker_id}_avg.npy", avg)
    np.save(db_dir / f"{speaker_id}_samples.npy", samples)

    db, _mod = _fresh_db(db_dir)

    assert db.list_speakers() == [
        {
            "id": speaker_id,
            "name": "Legacy",
            "sample_count": 2,
            "sample_spread": None,
            "created_at": "2026-04-01T00:00:00",
            "updated_at": "2026-04-02T00:00:00",
        }
    ]
    migrated_samples = db._conn.execute(
        "SELECT COUNT(*) FROM speaker_samples WHERE speaker_id = ?",
        (speaker_id,),
    ).fetchone()[0]
    assert migrated_samples == 2
    assert (db_dir / "index.json.migrated.bak").exists()


def test_legacy_voiceprint_migration_skips_unreadable_index(tmp_path):
    db_dir = tmp_path / "voiceprints"
    db_dir.mkdir(parents=True)
    (db_dir / "index.json").write_text("{not-json", encoding="utf-8")

    db, _mod = _fresh_db(db_dir)

    assert db.list_speakers() == []
    assert (db_dir / "index.json").exists()


def test_legacy_voiceprint_migration_ignores_missing_avg_and_existing_db(tmp_path):
    db_dir = tmp_path / "voiceprints"
    db_dir.mkdir(parents=True)
    (db_dir / "index.json").write_text(
        json.dumps({"speakers": {"spk_missing": {"name": "Missing"}}}),
        encoding="utf-8",
    )

    db, _mod = _fresh_db(db_dir)
    assert db.list_speakers() == []
    assert (db_dir / "index.json.migrated.bak").exists()

    sid = db.add_speaker("Existing", _unit_vec(7100))
    (db_dir / "index.json").write_text(
        json.dumps({"speakers": {"spk_other": {"name": "Other"}}}),
        encoding="utf-8",
    )

    db._storage._maybe_migrate_legacy()

    assert [speaker["id"] for speaker in db.list_speakers()] == [sid]
    assert (db_dir / "index.json").exists()


def test_repository_crud_and_private_scan_edges(tmp_path):
    db, _mod = _fresh_db(tmp_path / "voiceprints")
    sid = db.add_speaker("Maple", _unit_vec(7200))

    db.rename_speaker(sid, "Maple Renamed")
    assert db.get_speaker(sid)["name"] == "Maple Renamed"
    assert db.get_speaker("spk_missing") is None

    repo = db._repository
    assert repo._find_best_match(_unit_vec(7200))[0] == sid
    assert repo._python_cosine_scan(np.zeros(256, dtype=np.float32)) == []

    with pytest.raises(ValueError, match="No samples"):
        repo._recompute_avg_and_spread("spk_no_samples")

    db.delete_speaker(sid)
    assert db.get_speaker(sid) is None
    with pytest.raises(ValueError, match="not found"):
        db.delete_speaker(sid)


def test_recompute_spread_handles_zero_average(tmp_path):
    db, _mod = _fresh_db(tmp_path / "voiceprints")
    zero = np.zeros(256, dtype=np.float32)
    sid = db.add_speaker("Zero", zero)

    db.update_speaker(sid, zero)

    row = db.get_speaker(sid)
    assert row["sample_count"] == 2
    assert row["sample_spread"] is None


def test_cohort_helpers_handle_paths_invalid_files_and_collectors(tmp_path):
    _fresh_voiceprint_module()
    cohort_mod = importlib.import_module("voiceprints.cohort")

    class DummyDB:
        _asnorm = None
        _cohort_generation = 3
        _cohort_built_gen = 0
        _lock = threading.RLock()

    manager = cohort_mod.VoiceprintCohortManager(
        DummyDB(),
        cohort_path=tmp_path / "configured.npy",
        embedding_dim=3,
    )

    assert manager.cohort_path == tmp_path / "configured.npy"
    assert manager.cohort_size == 0
    assert manager.resolve_path(save_path=tmp_path / "explicit.npy") == (
        tmp_path / "explicit.npy"
    )
    assert manager.resolve_path(transcriptions_dir=tmp_path) == (
        tmp_path / "configured.npy"
    )
    no_default = cohort_mod.VoiceprintCohortManager(DummyDB(), None, embedding_dim=3)
    assert no_default.resolve_path(transcriptions_dir=tmp_path) == (
        tmp_path / "asnorm_cohort.npy"
    )
    assert no_default.resolve_path() is None

    invalid_ndim = tmp_path / "invalid.npy"
    np.save(invalid_ndim, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    with pytest.raises(ValueError, match="Cohort must be 2D"):
        manager.load(str(invalid_ndim))
    assert manager._persisted_cohort_size(invalid_ndim) == 0

    corrupt = tmp_path / "corrupt.npy"
    corrupt.write_bytes(b"not-numpy")
    assert manager._persisted_cohort_size(corrupt) == 0
    assert manager._persisted_cohort_size(None) == 0

    assert manager._should_keep_existing_cohort(source_size=1, current_size=0) is False
    assert manager._should_keep_existing_cohort(source_size=1, current_size=2) is True
    assert (
        manager._should_keep_existing_cohort(
            source_size=1,
            current_size=cohort_mod.ASNORM_MIN_COHORT_SIZE,
        )
        is True
    )

    collected = []
    encoded = base64.b64encode(np.array([1, 2, 3], dtype=np.float32).tobytes()).decode()
    added = manager._collect_json_embeddings(
        payload={
            "speaker_embeddings": {
                "list": [1, 2, 3],
                "encoded": encoded,
                "wrong_shape": [1, 2],
                "ignored": {"bad": True},
            }
        },
        expected_shape=(3,),
        collected=collected,
    )
    assert added == 2
    assert len(collected) == 2

    result_path = tmp_path / "tr_collect" / "result.json"
    result_path.parent.mkdir()
    np.save(result_path.parent / "emb_good.npy", np.array([4, 5, 6], dtype=np.float32))
    np.save(result_path.parent / "emb_wrong.npy", np.array([1, 2], dtype=np.float32))
    (result_path.parent / "emb_bad.npy").write_bytes(b"bad")

    skipped = manager._collect_npy_embeddings(
        result_path=result_path,
        expected_shape=(3,),
        collected=collected,
    )

    assert skipped == 1
    assert len(collected) == 3


def test_lifespan_loads_saved_cohort_without_rebuild(tmp_path, monkeypatch):
    """Startup must load an existing cohort file instead of rebuilding it again."""
    transcriptions_dir = tmp_path / "transcriptions"
    transcriptions_dir.mkdir(parents=True, exist_ok=True)
    saved = np.stack([_unit_vec(21), _unit_vec(22)]).astype(np.float32)
    cohort_path = transcriptions_dir / "asnorm_cohort.npy"
    np.save(cohort_path, saved)

    voiceprint_mod = _fresh_main(monkeypatch, tmp_path)

    load_calls: list[str] = []
    original_load = voiceprint_mod.VoiceprintDB.load_cohort

    def _record_load(self, path: str, top_n: int = 200):
        load_calls.append(path)
        return original_load(self, path, top_n=top_n)

    def _fail_rebuild(self, transcriptions_dir: str, save_path: str | None = None):
        raise AssertionError("startup should load saved cohort instead of rebuilding")

    monkeypatch.setattr(voiceprint_mod.VoiceprintDB, "load_cohort", _record_load)
    monkeypatch.setattr(
        voiceprint_mod.VoiceprintDB,
        "build_cohort_from_transcriptions",
        _fail_rebuild,
    )

    sys.modules.pop("main", None)
    main_mod = importlib.import_module("main")

    from fastapi.testclient import TestClient

    with TestClient(main_mod.app) as client:
        assert load_calls == [str(cohort_path)]
        assert client.app.state.db.cohort_size == 2


def test_concurrent_upload_dedup_reuses_single_live_job(app_client, monkeypatch):
    """Two simultaneous uploads of the same bytes must dedup to one queued job."""
    transcriptions = importlib.import_module("api.routers.transcriptions")
    submission = importlib.import_module("application.transcription_submission")
    audio_infra = importlib.import_module("infra.audio")
    job_persistence = importlib.import_module("infra.job_persistence")
    job_runtime = importlib.import_module("infra.job_runtime")

    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    worker_calls: list[tuple[str, str | None]] = []

    def _fake_run_transcription(
        job_id,
        audio_path,
        language,
        min_speakers,
        max_speakers,
        pipeline,
        voiceprint_db,
        denoise_model=None,
        snr_threshold=None,
        file_hash=None,
        no_repeat_ngram_size=0,
    ):
        worker_calls.append((job_id, file_hash))
        started.set()
        assert release.wait(timeout=5), "test timed out waiting to release worker"

        submission.jobs[job_id]["status"] = "completed"
        submission.jobs[job_id]["result"] = {
            "id": job_id,
            "segments": [],
            "unique_speakers": [],
        }
        job_persistence._write_status(job_id, "completed", filename=audio_path.name)
        out_dir = transcriptions.TRANSCRIPTIONS_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(
            json.dumps({"id": job_id, "segments": [], "unique_speakers": []})
        )
        if file_hash:
            audio_infra.register_hash(file_hash, job_id)
            job_runtime.unregister_in_flight(file_hash)
        finished.set()

    monkeypatch.setattr(submission, "run_transcription", _fake_run_transcription)

    barrier = threading.Barrier(2)

    def _submit():
        barrier.wait()
        return app_client.post(
            "/api/transcribe",
            files={"file": ("same.wav", _wav_bytes(), "audio/wav")},
            data={"language": "en"},
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(_submit) for _ in range(2)]
        responses = [future.result(timeout=10) for future in futures]

    assert started.wait(timeout=1), "background worker never started"

    payloads = []
    for response in responses:
        assert response.status_code == 200, response.text
        payloads.append(response.json())

    ids = {payload["id"] for payload in payloads}
    assert len(ids) == 1, f"Concurrent dedup should reuse one job id, got {payloads}"
    assert {payload.get("status") for payload in payloads} == {"queued"}, payloads
    assert sum(payload.get("deduplicated") is True for payload in payloads) == 1, (
        "Exactly one concurrent requester should take the in-flight dedup path"
    )
    assert len(worker_calls) == 1, (
        "Concurrent dedup regression: more than one background transcription worker "
        f"started for the same upload: {worker_calls}"
    )

    release.set()
    assert finished.wait(timeout=2), "background worker did not finish cleanly"
