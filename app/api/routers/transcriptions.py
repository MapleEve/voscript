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

from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException
from fastapi import Path as FPath
from fastapi import Request, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from api.deps import get_db, get_pipeline
from application.transcription_submission import (
    TranscriptionSubmissionCommand,
    TranscriptionSubmissionError,
    submit_transcription_upload,
)
from application.transcription_records import (
    TranscriptionRecordError,
    build_export_payload,
    get_audio_artifact,
    get_job_status,
    list_transcriptions as list_transcription_records,
    load_transcription_result,
    reassign_speaker as reassign_transcription_speaker,
)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _raise_submission_http_error(exc: TranscriptionSubmissionError) -> None:
    status_code = 413 if exc.reason == "upload_too_large" else 503
    raise HTTPException(status_code, str(exc)) from exc


def _raise_record_http_error(exc: TranscriptionRecordError) -> None:
    status_codes = {
        "invalid_transcription_id": 400,
        "job_not_found": 404,
        "transcription_not_found": 404,
        "corrupt_result": 409,
        "missing_audio": 404,
        "invalid_speaker_id": 422,
        "missing_voiceprint": 404,
        "segment_not_found": 404,
        "unsupported_export_format": 400,
    }
    raise HTTPException(status_codes.get(exc.reason, 500), str(exc)) from exc


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
    denoise_model: str = Form(None),
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

    try:
        submission = await submit_transcription_upload(
            TranscriptionSubmissionCommand(
                file=file,
                pipeline=pipeline,
                voiceprint_db=voiceprint_db,
                language=language,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                denoise_model=denoise_model,
                snr_threshold=snr_threshold,
                no_repeat_ngram_size=no_repeat_ngram_size,
            ),
        )
    except TranscriptionSubmissionError as exc:
        _raise_submission_http_error(exc)

    response = {"id": submission.job_id, "status": submission.status}
    if submission.deduplicated:
        response["deduplicated"] = True
    return response


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: Annotated[str, FPath(pattern=r"^tr_[A-Za-z0-9_-]{1,64}$")],
):
    try:
        return get_job_status(job_id)
    except TranscriptionRecordError as exc:
        _raise_record_http_error(exc)


@router.get("/transcriptions")
async def list_transcriptions():
    return list_transcription_records()


@router.get("/transcriptions/{tr_id}")
async def get_transcription(
    tr_id: Annotated[str, FPath(pattern=r"^tr_[A-Za-z0-9_-]{1,64}$")],
):
    try:
        return load_transcription_result(tr_id)
    except TranscriptionRecordError as exc:
        _raise_record_http_error(exc)


@router.get("/transcriptions/{tr_id}/audio")
async def download_audio(
    tr_id: Annotated[str, FPath(pattern=r"^tr_[A-Za-z0-9_-]{1,64}$")],
):
    """Return the original uploaded audio file for this transcription."""
    try:
        audio = get_audio_artifact(tr_id)
    except TranscriptionRecordError as exc:
        _raise_record_http_error(exc)
    return FileResponse(audio.path, filename=audio.filename)


@router.put("/transcriptions/{tr_id}/segments/{seg_id}/speaker")
async def reassign_speaker(
    request: Request,
    tr_id: Annotated[str, FPath(pattern=r"^tr_[A-Za-z0-9_-]{1,64}$")],
    seg_id: int,
    speaker_name: str = Form(...),
    speaker_id: str = Form(None),
):
    voiceprint_db = get_db(request) if speaker_id else None
    try:
        return reassign_transcription_speaker(
            tr_id,
            seg_id,
            speaker_name,
            speaker_id,
            voiceprint_db=voiceprint_db,
        )
    except TranscriptionRecordError as exc:
        _raise_record_http_error(exc)


@router.get("/export/{tr_id}")
async def export_transcription(
    tr_id: Annotated[str, FPath(pattern=r"^tr_[A-Za-z0-9_-]{1,64}$")],
    format: str = "srt",
):
    try:
        payload = build_export_payload(tr_id, format)
    except TranscriptionRecordError as exc:
        _raise_record_http_error(exc)

    if payload.file_path is not None:
        return FileResponse(
            payload.file_path,
            media_type=payload.media_type,
            filename=payload.filename,
        )
    return PlainTextResponse(
        payload.text or "",
        media_type=payload.media_type,
        headers={"Content-Disposition": f'attachment; filename="{payload.filename}"'},
    )
