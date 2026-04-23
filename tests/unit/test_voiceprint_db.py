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
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

_APP_DIR = Path(__file__).resolve().parents[2] / "app"


def _fresh_voiceprint_module():
    sys.modules.pop("voiceprint_db", None)
    return importlib.import_module("voiceprint_db")


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
        if name in {"main", "config", "voiceprint_db"}:
            sys.modules.pop(name, None)
        elif name == "api" or name.startswith("api."):
            sys.modules.pop(name, None)
        elif name.startswith("services."):
            sys.modules.pop(name, None)

    voiceprint_mod = importlib.import_module("voiceprint_db")
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
    job_service = importlib.import_module("services.job_service")

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

        transcriptions.jobs[job_id]["status"] = "completed"
        transcriptions.jobs[job_id]["result"] = {
            "id": job_id,
            "segments": [],
            "unique_speakers": [],
        }
        transcriptions._write_status(job_id, "completed", filename=audio_path.name)
        out_dir = transcriptions.TRANSCRIPTIONS_DIR / job_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "result.json").write_text(
            json.dumps({"id": job_id, "segments": [], "unique_speakers": []})
        )
        if file_hash:
            job_service.register_hash(file_hash, job_id)
            job_service.unregister_in_flight(file_hash)
        finished.set()

    monkeypatch.setattr(transcriptions, "run_transcription", _fake_run_transcription)

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
