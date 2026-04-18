"""FastAPI service for voice transcription with speaker identification."""

import hmac
import json
import os
import subprocess
import uuid
import logging
from datetime import datetime
from pathlib import Path, PurePosixPath
from threading import Thread

from fastapi import FastAPI, File, Form, UploadFile, HTTPException, Request
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    PlainTextResponse,
)
from fastapi.staticfiles import StaticFiles

from pipeline import TranscriptionPipeline
from voiceprint_db import VoiceprintDB

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))
TRANSCRIPTIONS_DIR = DATA_DIR / "transcriptions"
UPLOADS_DIR = DATA_DIR / "uploads"
VOICEPRINTS_DIR = DATA_DIR / "voiceprints"

API_KEY = (os.getenv("API_KEY") or "").strip() or None
# Paths that must stay open even when API_KEY auth is enabled. "/" is the
# bundled web UI (browsers can't attach a Bearer header to a direct
# navigation — the UI's own fetch() calls to /api/* still carry the key).
# /static/* serves the UI's assets. /healthz is a liveness probe. /docs
# /redoc /openapi.json are FastAPI's auto docs.
# We match exact strings for everything except /static/ to avoid a
# startswith("/docs") bypass like /docsXYZ.
PUBLIC_EXACT_PATHS = {
    "/",
    "/healthz",
    "/docs",
    "/redoc",
    "/openapi.json",
}
PUBLIC_PATH_PREFIXES = ("/static/",)

# Cap how much any single upload can occupy on disk. Whisper + pyannote
# comfortably handle 2 GB of audio (~20 h @ typical bitrates); anything
# beyond that is either a mistake or an attempt to exhaust storage.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
UPLOAD_CHUNK = 1 << 20  # 1 MiB

for d in [TRANSCRIPTIONS_DIR, UPLOADS_DIR, VOICEPRINTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

if API_KEY is None:
    logger.warning(
        "API_KEY is not set. The service is accepting unauthenticated requests. "
        "Do not expose this port to untrusted networks."
    )
else:
    logger.info("API_KEY auth enabled for /api/* and / (Bearer or X-API-Key).")

app = FastAPI(title="Voice Transcribe", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if API_KEY is None:
        return await call_next(request)

    path = request.url.path
    if path in PUBLIC_EXACT_PATHS or any(
        path.startswith(p) for p in PUBLIC_PATH_PREFIXES
    ):
        return await call_next(request)

    auth_header = request.headers.get("authorization", "")
    bearer = ""
    if auth_header.lower().startswith("bearer "):
        bearer = auth_header.split(" ", 1)[1].strip()
    header_key = request.headers.get("x-api-key", "").strip()

    # Constant-time comparison — no timing signal leaks the key prefix.
    if not (
        hmac.compare_digest(bearer, API_KEY) or hmac.compare_digest(header_key, API_KEY)
    ):
        return JSONResponse(
            {"detail": "Unauthorized"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )

    return await call_next(request)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


pipeline = TranscriptionPipeline()
voiceprint_db = VoiceprintDB(str(VOICEPRINTS_DIR))

# In-memory job status
jobs: dict[str, dict] = {}


def _convert_to_wav(input_path: Path) -> Path:
    """Convert any audio format to 16 kHz mono WAV via ffmpeg.

    We shell out to ffmpeg directly instead of using pydub because pydub's
    mediainfo_json() raises KeyError('codec_type') on newer ffmpeg output
    for some Opus/container combinations (see jiaaro/pydub#638). ffmpeg
    itself handles every format faster-whisper / pyannote ingest, so this
    is the simpler and more robust path.
    """
    wav_path = input_path.with_suffix(".wav")
    if input_path.suffix.lower() == ".wav":
        return input_path
    # "--" closes ffmpeg's option parsing so a filename like `-foo.mp4`
    # can't be interpreted as a flag. Defense in depth — the upload path
    # already strips client-side directory components and prefixes the
    # job_id, so input_path always starts with /data/uploads/tr_...
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(input_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-f",
            "wav",
            "--",
            str(wav_path),
        ],
        check=True,
    )
    return wav_path


def _run_transcription(
    job_id: str, audio_path: Path, language: str, min_speakers: int, max_speakers: int
):
    """Background transcription worker."""
    try:
        jobs[job_id]["status"] = "converting"
        wav_path = _convert_to_wav(audio_path)

        jobs[job_id]["status"] = "transcribing"
        result = pipeline.process(
            str(wav_path),
            language=language,
            min_speakers=min_speakers or None,
            max_speakers=max_speakers or None,
        )

        # Match speakers against voiceprint DB
        jobs[job_id]["status"] = "identifying"
        speaker_map = {}
        for spk_label, embedding in result["speaker_embeddings"].items():
            spk_id, spk_name, sim = voiceprint_db.identify(embedding)
            speaker_map[spk_label] = {
                "matched_id": spk_id,
                "matched_name": spk_name or spk_label,
                "similarity": round(sim, 4),
                "embedding_key": spk_label,
            }

        # Build final segments
        segments = []
        for i, seg in enumerate(result["segments"]):
            spk_label = seg["speaker"]
            match = speaker_map.get(spk_label, {})
            out = {
                "id": i,
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker_label": spk_label,
                "speaker_id": match.get("matched_id"),
                "speaker_name": match.get("matched_name", spk_label),
                "similarity": match.get("similarity", 0),
            }
            # Forward word-level timestamps when forced alignment produced them
            # (0.3.0+). Absent when the language has no alignment model or
            # alignment failed — clients must treat the key as optional.
            if seg.get("words"):
                out["words"] = seg["words"]
            segments.append(out)

        # Save transcription result
        tr = {
            "id": job_id,
            "filename": audio_path.name,
            "created_at": datetime.now().isoformat(),
            "status": "completed",
            "language": language,
            "segments": segments,
            "speaker_map": speaker_map,
            "unique_speakers": result["unique_speakers"],
        }

        tr_dir = TRANSCRIPTIONS_DIR / job_id
        tr_dir.mkdir(exist_ok=True)
        (tr_dir / "result.json").write_text(
            json.dumps(tr, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # Save raw embeddings for later enrollment
        import numpy as np

        for spk_label, emb in result["speaker_embeddings"].items():
            np.save(tr_dir / f"emb_{spk_label}.npy", emb)

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = tr
        logger.info(
            "Job %s completed: %d segments, %d speakers",
            job_id,
            len(segments),
            len(speaker_map),
        )

    except Exception as e:
        logger.exception("Job %s failed", job_id)
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


# --- Routes ---


@app.get("/", response_class=HTMLResponse)
async def index():
    return (Path("static/index.html")).read_text(encoding="utf-8")


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str = Form("zh"),
    min_speakers: int = Form(0),
    max_speakers: int = Form(0),
):
    job_id = f"tr_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    # Strip any directory component a client may have smuggled in via the
    # multipart filename (e.g. `../../etc/passwd`, `/tmp/evil`). We never
    # trust the client-provided filename as a path segment.
    safe_filename = PurePosixPath(file.filename or "upload").name or "upload"
    save_path = UPLOADS_DIR / f"{job_id}_{safe_filename}"

    # Streaming size-capped copy — refuse unbounded uploads that could
    # exhaust disk. Delete the partial artifact on overflow so we don't
    # leave a huge file behind.
    size = 0
    with open(save_path, "wb") as f:
        while True:
            chunk = file.file.read(UPLOAD_CHUNK)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                f.close()
                save_path.unlink(missing_ok=True)
                raise HTTPException(
                    413,
                    f"Upload exceeds MAX_UPLOAD_BYTES ({MAX_UPLOAD_BYTES} bytes)",
                )
            f.write(chunk)

    jobs[job_id] = {
        "status": "queued",
        "filename": safe_filename,
        "created_at": datetime.now().isoformat(),
    }
    thread = Thread(
        target=_run_transcription,
        args=(job_id, save_path, language, min_speakers, max_speakers),
    )
    thread.start()

    return {"id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    job = jobs[job_id]
    resp = {"id": job_id, "status": job["status"], "filename": job.get("filename")}
    if job["status"] == "completed":
        resp["result"] = job["result"]
    elif job["status"] == "failed":
        resp["error"] = job.get("error")
    return resp


@app.get("/api/transcriptions")
async def list_transcriptions():
    results = []
    for tr_dir in sorted(TRANSCRIPTIONS_DIR.iterdir(), reverse=True):
        result_file = tr_dir / "result.json"
        if result_file.exists():
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
    return results


@app.get("/api/transcriptions/{tr_id}")
async def get_transcription(tr_id: str):
    result_file = TRANSCRIPTIONS_DIR / tr_id / "result.json"
    if not result_file.exists():
        raise HTTPException(404, "Transcription not found")
    return json.loads(result_file.read_text(encoding="utf-8"))


@app.put("/api/transcriptions/{tr_id}/segments/{seg_id}/speaker")
async def reassign_speaker(
    tr_id: str, seg_id: int, speaker_name: str = Form(...), speaker_id: str = Form(None)
):
    """Reassign a segment to a different speaker and optionally enroll the voiceprint."""
    result_file = TRANSCRIPTIONS_DIR / tr_id / "result.json"
    if not result_file.exists():
        raise HTTPException(404, "Transcription not found")
    data = json.loads(result_file.read_text(encoding="utf-8"))

    seg = next((s for s in data["segments"] if s["id"] == seg_id), None)
    if seg is None:
        raise HTTPException(404, "Segment not found")

    seg["speaker_name"] = speaker_name
    if speaker_id:
        seg["speaker_id"] = speaker_id

    result_file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"ok": True}


@app.post("/api/voiceprints/enroll")
async def enroll_speaker(
    tr_id: str = Form(...),
    speaker_label: str = Form(...),
    speaker_name: str = Form(...),
    speaker_id: str = Form(None),
):
    """Enroll or update a voiceprint from a transcription's speaker embedding."""
    import numpy as np

    emb_path = TRANSCRIPTIONS_DIR / tr_id / f"emb_{speaker_label}.npy"
    if not emb_path.exists():
        raise HTTPException(404, "Embedding not found for this speaker label")
    embedding = np.load(emb_path)

    if speaker_id and voiceprint_db.get_speaker(speaker_id):
        voiceprint_db.update_speaker(speaker_id, embedding, name=speaker_name)
        return {"action": "updated", "speaker_id": speaker_id}
    else:
        new_id = voiceprint_db.add_speaker(speaker_name, embedding)
        return {"action": "created", "speaker_id": new_id}


@app.get("/api/voiceprints")
async def list_voiceprints():
    return voiceprint_db.list_speakers()


@app.delete("/api/voiceprints/{speaker_id}")
async def delete_voiceprint(speaker_id: str):
    try:
        voiceprint_db.delete_speaker(speaker_id)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"ok": True}


@app.put("/api/voiceprints/{speaker_id}/name")
async def rename_voiceprint(speaker_id: str, name: str = Form(...)):
    try:
        voiceprint_db.rename_speaker(speaker_id, name)
    except ValueError as e:
        raise HTTPException(404, str(e))
    return {"ok": True}


@app.get("/api/export/{tr_id}")
async def export_transcription(tr_id: str, format: str = "srt"):
    result_file = TRANSCRIPTIONS_DIR / tr_id / "result.json"
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
            headers={"Content-Disposition": f"attachment; filename={tr_id}.srt"},
        )
    elif format == "txt":
        lines = []
        for seg in segments:
            ts = _format_timestamp(seg["start"])
            lines.append(f"[{ts}] {seg['speaker_name']}: {seg['text']}")
        return PlainTextResponse(
            "\n".join(lines),
            media_type="text/plain",
            headers={"Content-Disposition": f"attachment; filename={tr_id}.txt"},
        )
    elif format == "json":
        return FileResponse(
            result_file, media_type="application/json", filename=f"{tr_id}.json"
        )
    else:
        raise HTTPException(400, "Unsupported format. Use: srt, txt, json")


def _format_srt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"
