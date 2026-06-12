"""Endpoint edge coverage for transcription and voiceprint routers."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np


def _seed_result(transcriptions_dir: Path, tr_id: str, *, filename: str = "audio.wav"):
    tr_dir = transcriptions_dir / tr_id
    tr_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": tr_id,
        "filename": filename,
        "created_at": "2026-04-25T00:00:00+00:00",
        "segments": [
            {
                "id": 1,
                "start": None,
                "end": float("nan"),
                "speaker_label": "SPEAKER_00",
                "speaker_name": "Maple\nInjected",
                "speaker_id": "spk_old",
                "text": "hello",
            },
            {
                "id": 2,
                "start": 61.25,
                "end": 62.0,
                "speaker_label": "SPEAKER_01",
                "speaker_name": "Guest",
                "speaker_id": None,
                "text": "world",
            },
        ],
        "unique_speakers": ["Maple\nInjected", "Guest"],
        "speaker_map": {},
    }
    result_path = tr_dir / "result.json"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return result_path


def test_transcription_job_status_fallback_paths(app_client):
    import api.routers.transcriptions as router

    router.jobs["tr_memory_done"] = {
        "status": "completed",
        "filename": "done.wav",
        "result": {"id": "tr_memory_done"},
    }
    router.jobs["tr_memory_failed"] = {
        "status": "failed",
        "filename": "failed.wav",
        "error": "boom",
    }

    done = app_client.get("/api/jobs/tr_memory_done")
    assert done.status_code == 200
    assert done.json()["result"] == {"id": "tr_memory_done"}

    failed = app_client.get("/api/jobs/tr_memory_failed")
    assert failed.status_code == 200
    assert failed.json()["error"] == "boom"

    completed_dir = router.TRANSCRIPTIONS_DIR / "tr_disk_done"
    completed_dir.mkdir(parents=True)
    (completed_dir / "status.json").write_text(
        json.dumps({"status": "completed", "filename": "disk.wav"}),
        encoding="utf-8",
    )
    (completed_dir / "result.json").write_text("{bad-json", encoding="utf-8")
    disk_done = app_client.get("/api/jobs/tr_disk_done")
    assert disk_done.status_code == 200
    assert disk_done.json()["result"] is None

    queued_dir = router.TRANSCRIPTIONS_DIR / "tr_disk_queued"
    queued_dir.mkdir(parents=True)
    (queued_dir / "status.json").write_text(
        json.dumps({"status": "queued", "filename": "queued.wav"}),
        encoding="utf-8",
    )
    disk_queued = app_client.get("/api/jobs/tr_disk_queued")
    assert disk_queued.status_code == 200
    assert disk_queued.json()["status"] == "failed"

    failed_dir = router.TRANSCRIPTIONS_DIR / "tr_disk_failed"
    failed_dir.mkdir(parents=True)
    (failed_dir / "status.json").write_text(
        json.dumps({"status": "failed", "filename": "failed.wav", "error": "bad"}),
        encoding="utf-8",
    )
    disk_failed = app_client.get("/api/jobs/tr_disk_failed")
    assert disk_failed.status_code == 200
    assert disk_failed.json()["error"] == "bad"


def test_transcription_list_audio_export_and_reassign_paths(app_client):
    import api.routers.transcriptions as router

    tr_id = "tr_route_edges"
    _seed_result(router.TRANSCRIPTIONS_DIR, tr_id, filename="route_audio.wav")
    bad_dir = router.TRANSCRIPTIONS_DIR / "tr_bad_listing"
    bad_dir.mkdir(parents=True)
    (bad_dir / "result.json").write_text("{bad-json", encoding="utf-8")

    listing = app_client.get("/api/transcriptions")
    assert listing.status_code == 200
    assert any(
        row["id"] == tr_id and row["segment_count"] == 2 for row in listing.json()
    )

    missing_audio = app_client.get(f"/api/transcriptions/{tr_id}/audio")
    assert missing_audio.status_code == 404

    (router.UPLOADS_DIR / "route_audio.wav").write_bytes(b"audio")
    audio = app_client.get(f"/api/transcriptions/{tr_id}/audio")
    assert audio.status_code == 200
    assert audio.content == b"audio"

    srt = app_client.get(f"/api/export/{tr_id}?format=srt")
    assert srt.status_code == 200
    assert "00:00:00,000 --> 00:00:00,000" in srt.text
    assert "[Maple Injected] hello" in srt.text

    txt = app_client.get(f"/api/export/{tr_id}?format=txt")
    assert txt.status_code == 200
    assert "[01:01] Guest: world" in txt.text

    exported_json = app_client.get(f"/api/export/{tr_id}?format=json")
    assert exported_json.status_code == 200
    assert exported_json.json()["id"] == tr_id

    unsupported = app_client.get(f"/api/export/{tr_id}?format=vtt")
    assert unsupported.status_code == 400

    invalid_id = app_client.put(
        f"/api/transcriptions/{tr_id}/segments/1/speaker",
        data={"speaker_name": "Maple", "speaker_id": "not-safe"},
    )
    assert invalid_id.status_code == 422

    class FakeDB:
        def __init__(self, found):
            self.found = found

        def get_speaker(self, speaker_id):
            return {"id": speaker_id} if self.found else None

    app_client.app.state.db = FakeDB(found=False)
    missing_voiceprint = app_client.put(
        f"/api/transcriptions/{tr_id}/segments/1/speaker",
        data={"speaker_name": "Maple", "speaker_id": "spk_missing"},
    )
    assert missing_voiceprint.status_code == 404

    app_client.app.state.db = FakeDB(found=True)
    updated = app_client.put(
        f"/api/transcriptions/{tr_id}/segments/1/speaker",
        data={"speaker_name": "Maple", "speaker_id": "spk_known"},
    )
    assert updated.status_code == 200

    cleared = app_client.put(
        f"/api/transcriptions/{tr_id}/segments/2/speaker",
        data={"speaker_name": "Maple"},
    )
    assert cleared.status_code == 200

    result = json.loads((router.TRANSCRIPTIONS_DIR / tr_id / "result.json").read_text())
    assert result["segments"][0]["speaker_id"] == "spk_known"
    assert result["segments"][1]["speaker_id"] is None
    assert result["unique_speakers"] == ["Maple"]

    missing_segment = app_client.put(
        f"/api/transcriptions/{tr_id}/segments/99/speaker",
        data={"speaker_name": "Nobody"},
    )
    assert missing_segment.status_code == 404


def test_transcribe_rejects_before_background_thread_when_admission_budget_full(
    app_client,
    monkeypatch,
):
    import api.routers.transcriptions as router
    import infra.job_runtime as job_runtime

    monkeypatch.setattr(router, "TRANSCRIPTION_MAX_ACTIVE_JOBS", 1)
    monkeypatch.setattr(job_runtime, "_active_job_ids", {"tr_busy"})
    monkeypatch.setattr(job_runtime, "_in_flight_hashes", {})

    started = []

    class FailingThread:
        def __init__(self, *args, **kwargs):
            started.append(("created", args, kwargs))

        def start(self):
            started.append(("started",))

    monkeypatch.setattr(router, "Thread", FailingThread)

    response = app_client.post(
        "/api/transcribe",
        files={"file": ("budget.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")},
        data={"language": "en"},
    )

    assert response.status_code == 503
    assert "active job budget" in response.text
    assert started == []


def test_transcribe_rejects_in_flight_budget_after_durable_bootstrap(
    app_client,
    monkeypatch,
):
    import api.routers.transcriptions as router
    import infra.job_runtime as job_runtime

    monkeypatch.setattr(job_runtime, "_active_job_ids", set())
    monkeypatch.setattr(router, "TRANSCRIPTION_MAX_IN_FLIGHT_JOBS", 1)
    monkeypatch.setattr(job_runtime, "_in_flight_hashes", {"sha256:busy": "tr_busy"})

    started = []

    class FailingThread:
        def __init__(self, *args, **kwargs):
            started.append(("created", args, kwargs))

        def start(self):
            started.append(("started",))

    monkeypatch.setattr(router, "Thread", FailingThread)

    response = app_client.post(
        "/api/transcribe",
        files={"file": ("busy.wav", b"RIFF\x00\x00\x00\x00WAVEfmt ", "audio/wav")},
        data={"language": "en"},
    )

    assert response.status_code == 503
    assert "in-flight job budget" in response.text
    assert started == []
    assert not list(router.TRANSCRIPTIONS_DIR.glob("tr_*"))
    assert not list(router.UPLOADS_DIR.glob("tr_*"))
    assert job_runtime.active_job_count() == 0


def test_transcribe_reuses_duplicate_in_flight_when_budgets_are_full(
    app_client,
    monkeypatch,
):
    import api.routers.transcriptions as router
    import infra.job_runtime as job_runtime

    audio = b"RIFF\x00\x00\x00\x00WAVEfmt duplicate"
    file_hash = hashlib.sha256(audio).hexdigest()
    monkeypatch.setattr(router, "TRANSCRIPTION_MAX_ACTIVE_JOBS", 1)
    monkeypatch.setattr(router, "TRANSCRIPTION_MAX_IN_FLIGHT_JOBS", 1)
    monkeypatch.setattr(job_runtime, "_active_job_ids", {"tr_existing"})
    monkeypatch.setattr(job_runtime, "_in_flight_hashes", {file_hash: "tr_existing"})

    started = []

    class FailingThread:
        def __init__(self, *args, **kwargs):
            started.append(("created", args, kwargs))

        def start(self):
            started.append(("started",))

    monkeypatch.setattr(router, "Thread", FailingThread)

    response = app_client.post(
        "/api/transcribe",
        files={"file": ("duplicate.wav", audio, "audio/wav")},
        data={"language": "en"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": "tr_existing",
        "status": "queued",
        "deduplicated": True,
    }
    assert started == []
    assert not list(router.TRANSCRIPTIONS_DIR.glob("tr_*"))
    assert not list(router.UPLOADS_DIR.glob("tr_*"))
    assert job_runtime.active_job_count() == 1


def test_voiceprint_management_routes(app_client):
    import api.routers.voiceprints as router

    class FakeDB:
        def __init__(self):
            self.speakers = {}
            self.updated = []
            self.deleted = []
            self.renamed = []
            self.cohort_path = None
            self.last_cohort_skipped = 3

        def add_speaker(self, name, embedding):
            assert embedding.shape == (3,)
            self.speakers["spk_new"] = {"id": "spk_new", "name": name}
            return "spk_new"

        def update_speaker(self, speaker_id, embedding, name=None):
            self.updated.append((speaker_id, name, tuple(embedding.tolist())))

        def list_speakers(self):
            return list(self.speakers.values())

        def get_speaker(self, speaker_id):
            return self.speakers.get(speaker_id)

        def delete_speaker(self, speaker_id):
            if speaker_id not in self.speakers:
                raise ValueError("missing speaker")
            self.deleted.append(speaker_id)
            self.speakers.pop(speaker_id)

        def rename_speaker(self, speaker_id, name):
            if speaker_id not in self.speakers:
                raise ValueError("missing speaker")
            self.renamed.append((speaker_id, name))
            self.speakers[speaker_id]["name"] = name

        def build_cohort_from_transcriptions(self, transcriptions_dir):
            assert transcriptions_dir == str(router.TRANSCRIPTIONS_DIR)
            return 7

    fake_db = FakeDB()
    app_client.app.state.db = fake_db

    invalid_tr_id = app_client.post(
        "/api/voiceprints/enroll",
        data={
            "tr_id": "../bad",
            "speaker_label": "SPEAKER_00",
            "speaker_name": "Maple",
        },
    )
    assert invalid_tr_id.status_code == 400
    assert "Invalid transcription ID format" in invalid_tr_id.text

    invalid_speaker_label = app_client.post(
        "/api/voiceprints/enroll",
        data={
            "tr_id": "tr_voiceprint",
            "speaker_label": "../bad",
            "speaker_name": "Maple",
        },
    )
    assert invalid_speaker_label.status_code == 400
    assert "Invalid speaker label" in invalid_speaker_label.text

    missing = app_client.post(
        "/api/voiceprints/enroll",
        data={
            "tr_id": "tr_voiceprint",
            "speaker_label": "SPEAKER_00",
            "speaker_name": "Maple",
        },
    )
    assert missing.status_code == 404

    tr_dir = router.safe_tr_dir("tr_voiceprint")
    tr_dir.mkdir(parents=True, exist_ok=True)
    np.save(tr_dir / "emb_SPEAKER_00.npy", np.array([1.0, 2.0, 3.0], dtype=np.float32))

    created = app_client.post(
        "/api/voiceprints/enroll",
        data={
            "tr_id": "tr_voiceprint",
            "speaker_label": "SPEAKER_00",
            "speaker_name": "Maple",
        },
    )
    assert created.status_code == 200
    assert created.json() == {"action": "created", "speaker_id": "spk_new"}

    updated = app_client.post(
        "/api/voiceprints/enroll",
        data={
            "tr_id": "tr_voiceprint",
            "speaker_label": "SPEAKER_00",
            "speaker_name": "Maple Updated",
            "speaker_id": "spk_new",
        },
    )
    assert updated.status_code == 200
    assert updated.json() == {"action": "updated", "speaker_id": "spk_new"}
    assert fake_db.updated == [("spk_new", "Maple Updated", (1.0, 2.0, 3.0))]

    listing = app_client.get("/api/voiceprints")
    assert listing.status_code == 200
    assert listing.json() == [{"id": "spk_new", "name": "Maple"}]

    found = app_client.get("/api/voiceprints/spk_new")
    assert found.status_code == 200
    assert found.json()["name"] == "Maple"

    missing_get = app_client.get("/api/voiceprints/spk_missing")
    assert missing_get.status_code == 404

    renamed = app_client.put("/api/voiceprints/spk_new/name", data={"name": "Renamed"})
    assert renamed.status_code == 200
    assert fake_db.renamed == [("spk_new", "Renamed")]

    missing_rename = app_client.put(
        "/api/voiceprints/spk_missing/name", data={"name": "Missing"}
    )
    assert missing_rename.status_code == 404

    cohort = app_client.post("/api/voiceprints/rebuild-cohort")
    assert cohort.status_code == 200
    assert cohort.json()["cohort_size"] == 7
    assert cohort.json()["skipped"] == 3
    assert cohort.json()["saved_to"].endswith("asnorm_cohort.npy")

    deleted = app_client.delete("/api/voiceprints/spk_new")
    assert deleted.status_code == 200
    assert fake_db.deleted == ["spk_new"]

    missing_delete = app_client.delete("/api/voiceprints/spk_missing")
    assert missing_delete.status_code == 404
