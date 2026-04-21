"""Transcription endpoints.

Covers:
  POST   /api/transcribe
  GET    /api/jobs/{job_id}
  GET    /api/transcriptions
  GET    /api/transcriptions/{tr_id}
  GET    /api/transcriptions/{tr_id}/audio
  PUT    /api/transcriptions/{tr_id}/segments/{seg_id}/speaker
  GET    /api/export/{tr_id}
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path, PurePosixPath
from threading import Thread
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException
from fastapi import Path as FPath
from fastapi import Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from api.deps import get_db, get_pipeline
from config import MAX_UPLOAD_BYTES, TRANSCRIPTIONS_DIR, UPLOAD_CHUNK, UPLOADS_DIR
from services.audio_service import (
    lookup_hash,
    safe_log_filename,
    safe_tr_dir,
    save_upload_and_hash,
)
from services.job_service import jobs, run_transcription

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_srt_time(seconds: float) -> str:
    # [CQ-M13] 防御 None / NaN / 负秒——SRT 不允许负时间戳，NaN 会导致 int() 抛异常。
    if seconds is None or seconds != seconds:  # NaN 自身不等于自身
        seconds = 0.0
    seconds = max(0.0, float(seconds))
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp(seconds: float) -> str:
    if seconds is None or seconds != seconds:
        seconds = 0.0
    seconds = max(0.0, float(seconds))
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/transcribe")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    language: str = Form(None),
    min_speakers: int = Form(0),
    max_speakers: int = Form(0),
    denoise_model: str = Form("none"),
    snr_threshold: float = Form(None),
    no_repeat_ngram_size: str = Form("0"),
):
    try:
        no_repeat_ngram_size = int(no_repeat_ngram_size)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=422,
            detail=[
                {
                    "loc": ["body", "no_repeat_ngram_size"],
                    "msg": "value is not a valid integer",
                    "type": "type_error.integer",
                }
            ],
        )
    pipeline = get_pipeline(request)
    voiceprint_db = get_db(request)

    # Normalise empty string to None so pipeline treats it as auto-detect.
    language = language.strip() if language else None

    job_id = f"tr_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    safe_filename = PurePosixPath(file.filename or "upload").name or "upload"
    # Strip control chars before using the name in paths/logs — PurePosixPath.name
    # preserves newlines and ANSI escapes which would otherwise enable log injection.
    safe_filename = safe_log_filename(safe_filename) or "upload"
    save_path = UPLOADS_DIR / f"{job_id}_{safe_filename}"

    # PERF-C2: async write + streaming SHA-256 — no event-loop blockage on large uploads.
    try:
        _size, file_hash = await save_upload_and_hash(
            file, save_path, MAX_UPLOAD_BYTES, UPLOAD_CHUNK
        )
    except ValueError as exc:
        save_path.unlink(missing_ok=True)
        raise HTTPException(413, str(exc)) from exc

    # Dedup: if identical audio was already transcribed, return existing result.
    existing_id = lookup_hash(file_hash)
    if existing_id:
        save_path.unlink(missing_ok=True)
        logger.info(
            "Dedup hit: %s already transcribed as %s", safe_filename, existing_id
        )
        return {"id": existing_id, "status": "completed", "deduplicated": True}

    jobs[job_id] = {
        "status": "queued",
        "filename": safe_filename,
        "created_at": datetime.now().isoformat(),
    }
    # CD-C3: daemon=True ensures this thread does not prevent the process from
    # exiting on SIGTERM — the OS will clean up in-progress transcriptions on
    # shutdown rather than hanging indefinitely waiting for the thread to finish.
    thread = Thread(
        target=run_transcription,
        args=(
            job_id,
            save_path,
            language,
            min_speakers,
            max_speakers,
            pipeline,
            voiceprint_db,
            denoise_model,
            snr_threshold,
            file_hash,
            no_repeat_ngram_size if no_repeat_ngram_size >= 3 else 0,
        ),
        daemon=True,
    )
    thread.start()

    return {"id": job_id, "status": "queued"}


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: Annotated[str, FPath(pattern=r"^tr_[A-Za-z0-9_-]{1,64}$")],
):
    if job_id in jobs:
        job = jobs[job_id]
        resp = {"id": job_id, "status": job["status"], "filename": job.get("filename")}
        if job["status"] == "completed":
            resp["result"] = job["result"]
        elif job["status"] == "failed":
            resp["error"] = job.get("error")
        return resp

    # AR-C2 fallback: process restarted — try reading persisted status.json.
    status_path = TRANSCRIPTIONS_DIR / job_id / "status.json"
    result_path = TRANSCRIPTIONS_DIR / job_id / "result.json"

    if status_path.exists():
        try:
            status_data = json.loads(status_path.read_text())
        except Exception:
            raise HTTPException(404, "Job not found")

        current_status = status_data.get("status")

        if current_status == "completed" and result_path.exists():
            try:
                result = json.loads(result_path.read_text(encoding="utf-8"))
            except Exception:
                result = None
            return {
                "id": job_id,
                "status": "completed",
                "filename": status_data.get("filename"),
                "result": result,
            }

        if current_status not in ("completed", "failed"):
            # In-progress status persisted by a previous process that no longer
            # owns this job — treat as a restart failure.
            return {
                "id": job_id,
                "status": "failed",
                "error": "Process restarted while job was in progress",
                "filename": status_data.get("filename"),
            }

        return {
            "id": job_id,
            "status": current_status,
            "error": status_data.get("error"),
            "filename": status_data.get("filename"),
        }

    raise HTTPException(404, "Job not found")


@router.get("/transcriptions")
async def list_transcriptions():
    results = []
    for tr_dir in sorted(TRANSCRIPTIONS_DIR.iterdir(), reverse=True):
        if not tr_dir.is_dir():
            continue
        result_file = tr_dir / "result.json"
        if result_file.exists():
            try:
                data = json.loads(result_file.read_text(encoding="utf-8"))
                results.append(
                    {
                        "id": data["id"],
                        "filename": data["filename"],
                        "created_at": data["created_at"],
                        "segment_count": len(data["segments"]),
                        "speaker_count": len(data.get("unique_speakers", [])),
                    }
                )
            except Exception as exc:
                logger.warning(
                    "Skipping corrupt result.json in %s: %s", tr_dir.name, exc
                )
    return results


@router.get("/transcriptions/{tr_id}")
async def get_transcription(
    tr_id: Annotated[str, FPath(pattern=r"^tr_[A-Za-z0-9_-]{1,64}$")],
):
    result_file = safe_tr_dir(tr_id) / "result.json"
    if not result_file.exists():
        raise HTTPException(404, "Transcription not found")
    return json.loads(result_file.read_text(encoding="utf-8"))


@router.get("/transcriptions/{tr_id}/audio")
async def download_audio(
    tr_id: Annotated[str, FPath(pattern=r"^tr_[A-Za-z0-9_-]{1,64}$")],
):
    """Return the original uploaded audio file for this transcription."""
    result_file = safe_tr_dir(tr_id) / "result.json"
    if not result_file.exists():
        raise HTTPException(404, "Transcription not found")
    data = json.loads(result_file.read_text(encoding="utf-8"))
    audio_file = UPLOADS_DIR / data["filename"]
    if not audio_file.exists():
        raise HTTPException(404, "Original audio file not found")
    return FileResponse(audio_file, filename=data["filename"])


@router.put("/transcriptions/{tr_id}/segments/{seg_id}/speaker")
async def reassign_speaker(
    tr_id: Annotated[str, FPath(pattern=r"^tr_[A-Za-z0-9_-]{1,64}$")],
    seg_id: int,
    speaker_name: str = Form(...),
    speaker_id: str = Form(None),
):
    """Reassign a segment to a different speaker and optionally enroll the voiceprint."""
    result_file = safe_tr_dir(tr_id) / "result.json"
    if not result_file.exists():
        raise HTTPException(404, "Transcription not found")
    data = json.loads(result_file.read_text(encoding="utf-8"))

    seg = next((s for s in data["segments"] if s["id"] == seg_id), None)
    if seg is None:
        raise HTTPException(404, "Segment not found")

    seg["speaker_name"] = speaker_name
    if speaker_id:
        seg["speaker_id"] = speaker_id

    # [CQ-H7] 同步更新 speaker_map，保持人工纠错在整条记录内一致。
    # 原 segment 可能引用一个 speaker_label（如 "SPEAKER_01"），我们在 speaker_map
    # 的对应条目上更新 matched_name / matched_id，而不是改 key。
    spk_label = seg.get("speaker_label")
    speaker_map = data.get("speaker_map") or {}
    if spk_label and spk_label in speaker_map:
        speaker_map[spk_label]["matched_name"] = speaker_name
        if speaker_id:
            speaker_map[spk_label]["matched_id"] = speaker_id
        data["speaker_map"] = speaker_map

    result_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"ok": True}


@router.get("/export/{tr_id}")
async def export_transcription(
    tr_id: Annotated[str, FPath(pattern=r"^tr_[A-Za-z0-9_-]{1,64}$")],
    format: str = "srt",
):
    result_file = safe_tr_dir(tr_id) / "result.json"
    if not result_file.exists():
        raise HTTPException(404, "Transcription not found")
    data = json.loads(result_file.read_text(encoding="utf-8"))
    segments = data["segments"]

    if format == "srt":
        lines = []
        for i, seg in enumerate(segments, 1):
            start = _format_srt_time(seg["start"])
            end = _format_srt_time(seg["end"])
            lines.append(
                f"{i}\n{start} --> {end}\n[{seg['speaker_name']}] {seg['text']}\n"
            )
        return PlainTextResponse(
            "\n".join(lines),
            media_type="text/srt",
            headers={"Content-Disposition": f'attachment; filename="{tr_id}.srt"'},
        )
    elif format == "txt":
        lines = []
        for seg in segments:
            ts = _format_timestamp(seg["start"])
            lines.append(f"[{ts}] {seg['speaker_name']}: {seg['text']}")
        return PlainTextResponse(
            "\n".join(lines),
            media_type="text/plain",
            headers={"Content-Disposition": f'attachment; filename="{tr_id}.txt"'},
        )
    elif format == "json":
        return FileResponse(
            result_file, media_type="application/json", filename=f"{tr_id}.json"
        )
    else:
        raise HTTPException(400, "Unsupported format. Use: srt, txt, json")
