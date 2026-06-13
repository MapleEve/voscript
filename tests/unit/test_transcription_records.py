"""Focused tests for application-level transcription record usecases."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _settings(records, tmp_path: Path):
    transcriptions_dir = tmp_path / "transcriptions"
    uploads_dir = tmp_path / "uploads"
    transcriptions_dir.mkdir()
    uploads_dir.mkdir()
    return records.TranscriptionRecordSettings(
        transcriptions_dir=transcriptions_dir,
        uploads_dir=uploads_dir,
    )


def _seed_result(
    transcriptions_dir: Path,
    tr_id: str,
    *,
    filename: str = "audio.wav",
    raw_text: str | None = None,
) -> Path:
    tr_dir = transcriptions_dir / tr_id
    tr_dir.mkdir(parents=True, exist_ok=True)
    result_path = tr_dir / "result.json"
    if raw_text is not None:
        result_path.write_text(raw_text, encoding="utf-8")
        return result_path

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
                "speaker_name": "Maple\r\nInjected",
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
        "unique_speakers": ["Maple\r\nInjected", "Guest"],
        "speaker_map": {},
    }
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    return result_path


def test_job_status_recovery_uses_runtime_jobs_then_disk(tmp_path):
    from application import transcription_records as records

    settings = _settings(records, tmp_path)
    runtime_jobs = {
        "tr_memory_done": {
            "status": "completed",
            "filename": "done.wav",
            "result": {"id": "tr_memory_done"},
        },
        "tr_memory_failed": {
            "status": "failed",
            "filename": "failed.wav",
            "error": "boom",
        },
    }

    assert records.get_job_status(
        "tr_memory_done",
        settings=settings,
        runtime_jobs=runtime_jobs,
    ) == {
        "id": "tr_memory_done",
        "status": "completed",
        "filename": "done.wav",
        "result": {"id": "tr_memory_done"},
    }
    assert (
        records.get_job_status(
            "tr_memory_failed",
            settings=settings,
            runtime_jobs=runtime_jobs,
        )["error"]
        == "boom"
    )

    done_dir = settings.transcriptions_dir / "tr_disk_done"
    done_dir.mkdir()
    (done_dir / "status.json").write_text(
        json.dumps({"status": "completed", "filename": "disk.wav"}),
        encoding="utf-8",
    )
    (done_dir / "result.json").write_text("{bad-json", encoding="utf-8")
    disk_done = records.get_job_status("tr_disk_done", settings=settings)
    assert disk_done["status"] == "completed"
    assert disk_done["result"] is None

    queued_dir = settings.transcriptions_dir / "tr_disk_queued"
    queued_dir.mkdir()
    (queued_dir / "status.json").write_text(
        json.dumps({"status": "queued", "filename": "queued.wav"}),
        encoding="utf-8",
    )
    disk_queued = records.get_job_status("tr_disk_queued", settings=settings)
    assert disk_queued == {
        "id": "tr_disk_queued",
        "status": "failed",
        "error": "Process restarted while job was in progress",
        "filename": "queued.wav",
    }

    corrupt_dir = settings.transcriptions_dir / "tr_badstatus"
    corrupt_dir.mkdir()
    (corrupt_dir / "status.json").write_text("{not-json", encoding="utf-8")
    with pytest.raises(records.TranscriptionRecordError) as exc_info:
        records.get_job_status("tr_badstatus", settings=settings)
    assert exc_info.value.reason == "job_not_found"
    assert str(exc_info.value) == "Job not found"


def test_record_listing_artifact_audio_and_exports(tmp_path):
    from application import transcription_records as records

    settings = _settings(records, tmp_path)
    tr_id = "tr_record_edges"
    _seed_result(settings.transcriptions_dir, tr_id, filename="route_audio.wav")
    _seed_result(settings.transcriptions_dir, "tr_corrupt", raw_text="{bad-json")

    listing = records.list_transcriptions(settings=settings)
    assert [row for row in listing if row["id"] == tr_id and row["segment_count"] == 2]

    with pytest.raises(records.TranscriptionRecordError) as missing_audio:
        records.get_audio_artifact(tr_id, settings=settings)
    assert missing_audio.value.reason == "missing_audio"
    assert str(missing_audio.value) == "Original audio file not found"

    (settings.uploads_dir / "route_audio.wav").write_bytes(b"audio")
    audio = records.get_audio_artifact(tr_id, settings=settings)
    assert audio.path == settings.uploads_dir / "route_audio.wav"
    assert audio.filename == "route_audio.wav"

    srt = records.build_export_payload(tr_id, "srt", settings=settings)
    assert srt.text is not None
    assert srt.file_path is None
    assert srt.media_type == "text/srt"
    assert srt.filename == f"{tr_id}.srt"
    assert "00:00:00,000 --> 00:00:00,000" in srt.text
    assert "[Maple Injected] hello" in srt.text

    txt = records.build_export_payload(tr_id, "txt", settings=settings)
    assert txt.text == "[00:00] Maple Injected: hello\n[01:01] Guest: world"
    assert txt.media_type == "text/plain"

    exported_json = records.build_export_payload(tr_id, "json", settings=settings)
    assert exported_json.text is None
    assert (
        exported_json.file_path == settings.transcriptions_dir / tr_id / "result.json"
    )
    assert exported_json.media_type == "application/json"
    assert exported_json.filename == f"{tr_id}.json"

    with pytest.raises(records.TranscriptionRecordError) as unsupported:
        records.build_export_payload(tr_id, "vtt", settings=settings)
    assert unsupported.value.reason == "unsupported_export_format"
    assert str(unsupported.value) == "Unsupported format. Use: srt, txt, json"

    for operation in (
        records.load_transcription_result,
        records.get_audio_artifact,
        lambda bad_id, *, settings: records.build_export_payload(
            bad_id,
            "txt",
            settings=settings,
        ),
    ):
        with pytest.raises(records.TranscriptionRecordError) as corrupt:
            operation("tr_corrupt", settings=settings)
        assert corrupt.value.reason == "corrupt_result"
        assert str(corrupt.value) == "Corrupt transcription artifact"


@pytest.mark.parametrize(
    "filename",
    [
        "../outside.wav",
        "/" + "outside.wav",
    ],
)
def test_audio_artifact_rejects_result_filename_that_escapes_uploads(
    tmp_path,
    filename,
):
    from application import transcription_records as records

    settings = _settings(records, tmp_path)
    tr_id = "tr_unsafe_audio_name"
    outside_audio = tmp_path / "outside.wav"
    outside_audio.write_bytes(b"outside")
    _seed_result(settings.transcriptions_dir, tr_id, filename=filename)

    with pytest.raises(records.TranscriptionRecordError) as exc_info:
        records.get_audio_artifact(tr_id, settings=settings)

    assert exc_info.value.reason == "corrupt_result"
    assert str(exc_info.value) == "Corrupt transcription artifact"


def test_speaker_reassignment_validates_voiceprint_and_updates_result(tmp_path):
    from application import transcription_records as records

    settings = _settings(records, tmp_path)
    tr_id = "tr_speaker_edges"
    _seed_result(settings.transcriptions_dir, tr_id)

    class FakeDB:
        def __init__(self, found: bool) -> None:
            self.found = found

        def get_speaker(self, speaker_id):
            return {"id": speaker_id} if self.found else None

    with pytest.raises(records.TranscriptionRecordError) as invalid_id:
        records.reassign_speaker(
            tr_id,
            1,
            "Maple",
            "not-safe",
            voiceprint_db=FakeDB(found=True),
            settings=settings,
        )
    assert invalid_id.value.reason == "invalid_speaker_id"
    assert str(invalid_id.value) == "Invalid speaker_id format"

    with pytest.raises(records.TranscriptionRecordError) as missing_voiceprint:
        records.reassign_speaker(
            tr_id,
            1,
            "Maple",
            "spk_missing",
            voiceprint_db=FakeDB(found=False),
            settings=settings,
        )
    assert missing_voiceprint.value.reason == "missing_voiceprint"
    assert str(missing_voiceprint.value) == "Voiceprint spk_missing not found"

    assert records.reassign_speaker(
        tr_id,
        1,
        "Maple",
        "spk_known",
        voiceprint_db=FakeDB(found=True),
        settings=settings,
    ) == {"ok": True}
    assert records.reassign_speaker(
        tr_id,
        2,
        "Maple",
        None,
        settings=settings,
    ) == {"ok": True}

    data = json.loads((settings.transcriptions_dir / tr_id / "result.json").read_text())
    assert data["segments"][0]["speaker_id"] == "spk_known"
    assert data["segments"][1]["speaker_id"] is None
    assert data["unique_speakers"] == ["Maple"]

    with pytest.raises(records.TranscriptionRecordError) as missing_segment:
        records.reassign_speaker(tr_id, 99, "Nobody", settings=settings)
    assert missing_segment.value.reason == "segment_not_found"
    assert str(missing_segment.value) == "Segment not found"


def test_transcription_records_module_stays_out_of_api_ring():
    from application import transcription_records as records

    source = Path(records.__file__).read_text(encoding="utf-8")

    assert "fastapi" not in source
    assert "HTTPException" not in source
    assert "UploadFile" not in source
    assert "api." not in source


def test_transcription_records_module_delegates_filesystem_details_to_infra():
    from application import transcription_records as records

    source = Path(records.__file__).read_text(encoding="utf-8")

    assert "from infra.transcription_records import" in source
    assert "json.loads" not in source
    assert "read_text(" not in source
    assert "write_text(" not in source
    assert "_atomic_write_json" not in source
    assert "iterdir(" not in source
    assert ' / "status.json"' not in source
    assert ' / "result.json"' not in source
    assert "PurePosixPath" not in source
    assert "PureWindowsPath" not in source
