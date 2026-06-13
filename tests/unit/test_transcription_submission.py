"""Focused tests for application-level transcription upload submission."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import infra.job_runtime as job_runtime


class MemoryUpload:
    def __init__(self, filename: str, content: bytes) -> None:
        self.filename = filename
        self._content = content
        self._offset = 0

    async def read(self, size: int = -1) -> bytes:
        if self._offset >= len(self._content):
            return b""
        if size is None or size < 0:
            size = len(self._content) - self._offset
        chunk = self._content[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def _submission_settings(
    *,
    uploads_dir: Path,
    transcriptions_dir: Path,
    min_free_disk_bytes: int = 0,
):
    from application import transcription_submission as submission

    return submission.TranscriptionSubmissionSettings(
        max_upload_bytes=1024,
        upload_chunk=8,
        max_active_jobs=2,
        max_in_flight_jobs=2,
        uploads_dir=uploads_dir,
        transcriptions_dir=transcriptions_dir,
        min_free_disk_bytes=min_free_disk_bytes,
        denoise_max_audio_duration_sec=111.0,
        embedding_preload_max_audio_duration_sec=222.0,
        whisperx_align_max_audio_duration_sec=333.0,
    )


async def _write_upload(file, save_path, max_upload_bytes, upload_chunk):
    del max_upload_bytes
    sha256 = hashlib.sha256()
    size = 0
    with save_path.open("wb") as handle:
        while chunk := await file.read(upload_chunk):
            size += len(chunk)
            handle.write(chunk)
            sha256.update(chunk)
    return size, sha256.hexdigest()


def test_submit_transcription_upload_retains_failed_record_on_thread_start_failure(
    tmp_path,
    monkeypatch,
):
    """Thread-start failure cleans transient state but keeps durable failure."""
    from application import transcription_submission as submission
    import infra.job_persistence as job_persistence

    transcriptions_dir = tmp_path / "transcriptions"
    uploads_dir = tmp_path / "uploads"
    transcriptions_dir.mkdir()
    uploads_dir.mkdir()

    monkeypatch.setattr(job_runtime, "_active_job_ids", set())
    monkeypatch.setattr(job_runtime, "_in_flight_hashes", {})
    monkeypatch.setattr(job_persistence, "TRANSCRIPTIONS_DIR", transcriptions_dir)

    audio = b"RIFF\x00\x00\x00\x00WAVEfmt thread-start"
    file_hash = hashlib.sha256(audio).hexdigest()
    started = []

    class FailingThread:
        def __init__(self, *args, **kwargs):
            started.append(("created", args, kwargs))

        def start(self):
            started.append(("started",))
            raise RuntimeError("thread boom")

    def worker(*args, **kwargs):
        raise AssertionError("worker should not run synchronously")

    async def upload_saver(file, save_path, max_upload_bytes, upload_chunk):
        del max_upload_bytes
        sha256 = hashlib.sha256()
        size = 0
        with save_path.open("wb") as handle:
            while chunk := await file.read(upload_chunk):
                size += len(chunk)
                handle.write(chunk)
                sha256.update(chunk)
        return size, sha256.hexdigest()

    monkeypatch.setattr(submission, "Thread", FailingThread)
    monkeypatch.setattr(submission, "run_transcription", worker)

    try:
        asyncio.run(
            submission.submit_transcription_upload(
                submission.TranscriptionSubmissionCommand(
                    file=MemoryUpload("../-y\nattack.wav", audio),
                    pipeline=object(),
                    voiceprint_db=object(),
                    language="en",
                    min_speakers=0,
                    max_speakers=0,
                    denoise_model=None,
                    snr_threshold=None,
                    no_repeat_ngram_size=2,
                ),
                settings=submission.TranscriptionSubmissionSettings(
                    max_upload_bytes=1024,
                    upload_chunk=8,
                    max_active_jobs=2,
                    max_in_flight_jobs=2,
                    uploads_dir=uploads_dir,
                    transcriptions_dir=transcriptions_dir,
                ),
                job_id_factory=lambda: "tr_submit",
                upload_saver=upload_saver,
            )
        )
    except submission.TranscriptionSubmissionError as exc:
        submission_error = exc
    else:
        raise AssertionError("submit_transcription_upload should fail thread startup")

    assert submission_error.reason == "thread_start_failed"
    assert (
        str(submission_error)
        == "Failed to start background transcription — retry later"
    )
    assert [entry[0] for entry in started] == ["created", "started"]
    created_kwargs = started[0][2]
    thread_args = created_kwargs["args"]
    assert created_kwargs["target"] is worker
    assert created_kwargs["daemon"] is True
    assert thread_args[0] == "tr_submit"
    assert thread_args[-1] == 0

    assert "tr_submit" in submission.jobs
    assert submission.jobs["tr_submit"]["status"] == "failed"
    assert (
        submission.jobs["tr_submit"]["error"]
        == "Failed to start background transcription"
    )
    assert list(uploads_dir.iterdir()) == []
    assert job_runtime.lookup_in_flight(file_hash) is None
    assert job_runtime.active_job_count() == 0

    status_path = transcriptions_dir / "tr_submit" / "status.json"
    assert status_path.exists()
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["status"] == "failed"
    assert status["filename"].startswith("-y")
    assert "\n" not in status["filename"]


def test_submit_transcription_upload_rejects_data_disk_pressure_before_bootstrap(
    tmp_path,
    monkeypatch,
):
    from application import transcription_submission as submission

    transcriptions_dir = tmp_path / "transcriptions"
    uploads_dir = tmp_path / "uploads"
    transcriptions_dir.mkdir()
    uploads_dir.mkdir()

    monkeypatch.setattr(job_runtime, "_active_job_ids", set())
    monkeypatch.setattr(job_runtime, "_in_flight_hashes", {})
    monkeypatch.setattr(job_runtime, "jobs", job_runtime._LRUJobsDict(maxsize=200))

    started = []
    status_writes = []
    audio = b"RIFF pressure"
    file_hash = hashlib.sha256(audio).hexdigest()

    class RecordingThread:
        def __init__(self, *args, **kwargs):
            started.append(("created", args, kwargs))

        def start(self):
            started.append(("started",))

    try:
        asyncio.run(
            submission.submit_transcription_upload(
                submission.TranscriptionSubmissionCommand(
                    file=MemoryUpload("pressure.wav", audio),
                    pipeline=object(),
                    voiceprint_db=object(),
                ),
                settings=_submission_settings(
                    uploads_dir=uploads_dir,
                    transcriptions_dir=transcriptions_dir,
                    min_free_disk_bytes=1024,
                ),
                job_id_factory=lambda: "tr_pressure",
                thread_factory=RecordingThread,
                upload_saver=_write_upload,
                status_writer=lambda *args, **kwargs: (
                    status_writes.append((args, kwargs)) or True
                ),
                disk_usage=lambda path: SimpleNamespace(free=1023),
                audio_duration_reader=lambda path: 12.5,
            )
        )
    except submission.TranscriptionSubmissionError as exc:
        submission_error = exc
    else:
        raise AssertionError("submit_transcription_upload should reject disk pressure")

    assert submission_error.reason == "data_disk_pressure"
    assert "data disk free space" in str(submission_error)
    assert list(uploads_dir.iterdir()) == []
    assert list(transcriptions_dir.iterdir()) == []
    assert status_writes == []
    assert started == []
    assert "tr_pressure" not in submission.jobs
    assert submission.release_transcription_admission("tr_pressure") is False
    assert submission.find_in_flight_transcription(file_hash) is None


def test_submit_transcription_upload_records_admission_snapshot(
    tmp_path,
    monkeypatch,
):
    from application import transcription_submission as submission

    transcriptions_dir = tmp_path / "transcriptions"
    uploads_dir = tmp_path / "uploads"
    transcriptions_dir.mkdir()
    uploads_dir.mkdir()

    monkeypatch.setattr(job_runtime, "_active_job_ids", set())
    monkeypatch.setattr(job_runtime, "_in_flight_hashes", {})
    monkeypatch.setattr(job_runtime, "jobs", job_runtime._LRUJobsDict(maxsize=200))

    started = []

    class RecordingThread:
        def __init__(self, *args, **kwargs):
            started.append(("created", args, kwargs))

        def start(self):
            started.append(("started",))

    result = asyncio.run(
        submission.submit_transcription_upload(
            submission.TranscriptionSubmissionCommand(
                file=MemoryUpload("snapshot.wav", b"RIFF snapshot"),
                pipeline=object(),
                voiceprint_db=object(),
            ),
            settings=_submission_settings(
                uploads_dir=uploads_dir,
                transcriptions_dir=transcriptions_dir,
                min_free_disk_bytes=1024,
            ),
            job_id_factory=lambda: "tr_snapshot",
            thread_factory=RecordingThread,
            upload_saver=_write_upload,
            status_writer=lambda *args, **kwargs: True,
            disk_usage=lambda path: SimpleNamespace(free=2048),
            audio_duration_reader=lambda path: 42.25,
        )
    )

    assert result.job_id == "tr_snapshot"
    assert result.status == "queued"
    assert [entry[0] for entry in started] == ["created", "started"]

    admission = submission.jobs["tr_snapshot"]["admission"]
    assert admission["active_jobs"] == 0
    assert admission["in_flight_jobs"] == 0
    assert admission["data_disk"] == {
        "free_bytes": 2048,
        "min_free_bytes": 1024,
    }
    assert admission["memory_sensitive_stage_limits"] == {
        "DENOISE_MAX_AUDIO_DURATION_SEC": 111.0,
        "EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC": 222.0,
        "WHISPERX_ALIGN_MAX_AUDIO_DURATION_SEC": 333.0,
    }
    assert admission["audio_duration_seconds"] == 42.25


def test_transcription_submission_module_stays_out_of_api_ring():
    from application import transcription_submission as submission

    source = Path(submission.__file__).read_text(encoding="utf-8")

    assert "fastapi" not in source
    assert "HTTPException" not in source
    assert "UploadFile" not in source
    assert "api." not in source
