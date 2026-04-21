"""TEST-C3: _run_transcription job-service state machine & persistence.

Exercises:
- run_transcription writes status.json to disk after a successful run (AR-C2)
- recover_orphan_jobs() flips leftover in-progress statuses to ``failed``
- the in-memory jobs dict is a bounded LRU that evicts the oldest entry past
  ``maxsize`` (CQ-H2 / PERF-C1)

Pipeline and voiceprint_db are full mocks — no Whisper, no pyannote, no sqlite.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "app"


# ---------------------------------------------------------------------------
# Fixture: isolated job_service module with DATA_DIR redirected to tmp_path
# ---------------------------------------------------------------------------


@pytest.fixture
def job_service(tmp_path, monkeypatch):
    """Import job_service freshly with DATA_DIR pointed at *tmp_path*.

    The module reads TRANSCRIPTIONS_DIR at import time, so we must evict any
    cached copy of config + job_service before re-importing. We also
    monkeypatch the heavy helpers (convert_to_wav, maybe_denoise) so the
    worker never shells out to ffmpeg.
    """
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.chdir(_APP_DIR)

    for _m in list(sys.modules):
        if _m in ("main", "config") or _m.startswith("api.") or _m == "api":
            sys.modules.pop(_m, None)
        elif _m.startswith("services."):
            sys.modules.pop(_m, None)

    mod = importlib.import_module("services.job_service")

    # Replace heavy audio helpers with identity / no-op stubs.
    monkeypatch.setattr(mod, "convert_to_wav", lambda p: p)
    monkeypatch.setattr(mod, "maybe_denoise", lambda p, *a, **kw: p)
    monkeypatch.setattr(mod, "register_hash", lambda *a, **kw: None)

    return mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_pipeline():
    """Return a MagicMock pipeline that produces one segment + one embedding."""
    pipeline = MagicMock()
    pipeline.process.return_value = {
        "segments": [
            {
                "start": 0.0,
                "end": 1.5,
                "text": "hello world",
                "speaker": "SPEAKER_00",
            }
        ],
        "language": "en",
        "speaker_embeddings": {
            "SPEAKER_00": np.zeros(256, dtype=np.float32),
        },
        "unique_speakers": ["SPEAKER_00"],
    }
    return pipeline


def _fake_voiceprint_db():
    """Return a mock VoiceprintDB — identify always misses (new speaker)."""
    db = MagicMock()
    db.identify.return_value = (None, None, 0.0)
    db._asnorm = None  # triggers the rebuild branch exactly once
    db.build_cohort_from_transcriptions.return_value = 0
    return db


# ---------------------------------------------------------------------------
# TEST-C3 – status.json written to disk
# ---------------------------------------------------------------------------


def test_job_status_written_to_disk(job_service, tmp_path):
    """A completed run must leave behind status.json with status=completed."""
    job_id = "tr_status_probe"
    job_service.jobs[job_id] = {"status": "queued", "filename": "probe.wav"}

    # Touch the fake audio file so Path(audio_path).name works downstream.
    audio_path = tmp_path / "uploads" / f"{job_id}_probe.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"")

    job_service.run_transcription(
        job_id=job_id,
        audio_path=audio_path,
        language="en",
        min_speakers=0,
        max_speakers=0,
        pipeline=_fake_pipeline(),
        voiceprint_db=_fake_voiceprint_db(),
    )

    status_path = tmp_path / "transcriptions" / job_id / "status.json"
    assert (
        status_path.exists()
    ), f"status.json missing at {status_path} — AR-C2 persistence broken"
    data = json.loads(status_path.read_text())
    assert data["status"] == "completed", f"expected completed, got {data!r}"
    assert data.get("error") is None

    # In-memory state should agree.
    assert job_service.jobs[job_id]["status"] == "completed"

    # result.json must also be written.
    result_path = tmp_path / "transcriptions" / job_id / "result.json"
    assert result_path.exists()
    result = json.loads(result_path.read_text())
    assert result["id"] == job_id
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "hello world"


# ---------------------------------------------------------------------------
# TEST-C3 – orphan recovery
# ---------------------------------------------------------------------------


def test_orphan_job_marked_failed_on_recover(job_service, tmp_path):
    """recover_orphan_jobs() must flip in-progress status.json to failed."""
    job_id = "tr_orphan"
    status_dir = tmp_path / "transcriptions" / job_id
    status_dir.mkdir(parents=True, exist_ok=True)

    # Simulate a status written by a previous process mid-transcription.
    (status_dir / "status.json").write_text(
        json.dumps({"status": "transcribing", "updated_at": "2026-04-20T10:00:00"})
    )

    job_service.recover_orphan_jobs()

    data = json.loads((status_dir / "status.json").read_text())
    assert data["status"] == "failed", f"orphan must be marked failed, got {data!r}"
    assert "restarted" in (data.get("error") or "").lower()

    # A *completed* job must not be touched.
    done_id = "tr_done"
    done_dir = tmp_path / "transcriptions" / done_id
    done_dir.mkdir(parents=True, exist_ok=True)
    (done_dir / "status.json").write_text(
        json.dumps({"status": "completed", "updated_at": "2026-04-20T10:00:00"})
    )

    job_service.recover_orphan_jobs()

    data = json.loads((done_dir / "status.json").read_text())
    assert data["status"] == "completed", "recover must leave completed jobs alone"


# ---------------------------------------------------------------------------
# TEST-C3 – bounded LRU job cache (CQ-H2 / PERF-C1)
# ---------------------------------------------------------------------------


def test_job_lru_eviction(job_service):
    """Exceeding maxsize must evict the oldest entry, not grow unboundedly."""
    LRUCls = job_service._LRUJobsDict
    cache = LRUCls(maxsize=3)

    cache["a"] = {"status": "queued"}
    cache["b"] = {"status": "queued"}
    cache["c"] = {"status": "queued"}
    assert "a" in cache and "b" in cache and "c" in cache

    # 4th insert — oldest ("a") must be evicted.
    cache["d"] = {"status": "queued"}
    assert "a" not in cache, "LRU must evict the oldest entry"
    assert all(k in cache for k in ("b", "c", "d"))

    # Touching an existing key moves it to the end → it survives the next
    # eviction round.
    cache["b"] = {"status": "updated"}
    cache["e"] = {"status": "queued"}
    assert "c" not in cache, "after touching 'b', 'c' is now the oldest"
    assert all(k in cache for k in ("b", "d", "e"))

    # get() returns default when missing, without raising.
    assert cache.get("missing") is None
    assert cache.get("missing", "sentinel") == "sentinel"
