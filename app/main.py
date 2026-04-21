"""FastAPI service for voice transcription with speaker identification."""

import hashlib
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
from fastapi.middleware.cors import CORSMiddleware
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


# Base cosine-similarity threshold for voiceprint identify(). The actual
# threshold per candidate is adaptive — see voiceprint_db.identify's docstring
# for the per-speaker relaxation rules.
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


VOICEPRINT_THRESHOLD = _env_float("VOICEPRINT_THRESHOLD", 0.75)

DENOISE_MODEL = os.getenv("DENOISE_MODEL", "none").strip().lower()

# SNR threshold (dB) below which DeepFilterNet is applied.
# Audio estimated at or above this level is considered clean and skipped,
# matching the A/B finding that DF hurts high-quality recordings (e.g. PLAUD Pin).
DENOISE_SNR_THRESHOLD = _env_float("DENOISE_SNR_THRESHOLD", 10.0)

# Lazy module-level handle so DeepFilterNet loads once at first use.
_df_model = None
_df_state = None


def _load_deepfilternet():
    global _df_model, _df_state
    if _df_model is None:
        import df as _df_pkg

        _df_model, _df_state, _ = _df_pkg.init_df()
        logger.info("DeepFilterNet model loaded")
    return _df_model, _df_state


def _estimate_snr(wav_path: Path) -> float:
    """Estimate signal-to-noise ratio (dB) using a simple energy-based heuristic.

    Strategy: divide the audio into short frames, compute per-frame RMS energy,
    then treat the bottom 20 % of frame energies as the noise floor and the top
    80 % as the speech signal.  SNR = 10 * log10(speech_power / noise_power).

    This is intentionally lightweight — no VAD model, no STFT — so it adds
    negligible latency before deciding whether to invoke DeepFilterNet.
    """
    import math
    import torchaudio

    waveform, sr = torchaudio.load(str(wav_path))
    # Flatten to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)  # shape: (num_samples,)

    # 30 ms frames
    frame_len = max(1, int(sr * 0.03))
    num_frames = len(waveform) // frame_len
    if num_frames < 5:
        # Too short to estimate reliably — assume clean
        return float("inf")

    frames = waveform[: num_frames * frame_len].reshape(num_frames, frame_len)
    frame_rms = frames.pow(2).mean(dim=1).sqrt()  # shape: (num_frames,)

    sorted_rms, _ = frame_rms.sort()
    noise_cutoff = max(1, int(num_frames * 0.20))
    noise_rms = sorted_rms[:noise_cutoff].mean().item()
    speech_rms = sorted_rms[noise_cutoff:].mean().item()

    if noise_rms < 1e-9:
        return float("inf")  # Silent noise floor — effectively infinite SNR

    snr_db = 10.0 * math.log10((speech_rms / noise_rms) ** 2)
    return snr_db


def _maybe_denoise(
    wav_path: Path, model: str = None, snr_threshold: float = None
) -> Path:
    """Return denoised WAV path if DENOISE_MODEL is set; otherwise return wav_path unchanged."""
    effective_model = (model or DENOISE_MODEL).strip().lower()
    if effective_model == "none":
        return wav_path

    threshold = snr_threshold if snr_threshold is not None else DENOISE_SNR_THRESHOLD
    out_path = wav_path.with_suffix(".denoised.wav")

    if effective_model == "deepfilternet":
        import torch, torchaudio

        snr_db = _estimate_snr(wav_path)
        if snr_db >= threshold:
            logger.info("DeepFilterNet skipped (SNR=%.1fdB, clean audio)", snr_db)
            return wav_path

        logger.info(
            "DeepFilterNet applying (SNR=%.1fdB < %.1fdB threshold)",
            snr_db,
            threshold,
        )
        model, df_state = _load_deepfilternet()
        import df as _df_pkg

        audio, sr = torchaudio.load(str(wav_path))
        if sr != df_state.sr():
            audio = torchaudio.functional.resample(audio, sr, df_state.sr())
        audio = audio.contiguous()
        with torch.backends.cudnn.flags(enabled=False):
            enhanced = _df_pkg.enhance(model, df_state, audio)
        torchaudio.save(
            str(out_path),
            enhanced.unsqueeze(0) if enhanced.dim() == 1 else enhanced,
            df_state.sr(),
        )
        logger.info("DeepFilterNet: denoised %s → %s", wav_path.name, out_path.name)

    elif effective_model == "noisereduce":
        import numpy as np, soundfile as sf, noisereduce as nr

        data, sr = sf.read(str(wav_path), dtype="float32")
        reduced = nr.reduce_noise(y=data, sr=sr, stationary=True)
        sf.write(str(out_path), reduced, sr)
        logger.info("noisereduce: denoised %s → %s", wav_path.name, out_path.name)

    else:
        logger.warning("Unknown DENOISE_MODEL=%r — skipping denoising", effective_model)
        return wav_path

    return out_path


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

_cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def require_api_key(request: Request, call_next):
    if API_KEY is None:
        return await call_next(request)

    # CORS preflight: CORSMiddleware handles the response headers; we must
    # not 401 before it runs.
    if request.method == "OPTIONS":
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

# Auto-build or load AS-norm cohort from existing transcriptions. This lets
# identify() use normalized scores instead of raw cosine against speaker-
# dependent baselines. Failure is non-fatal — we fall back to raw cosine.
try:
    _cohort_path = TRANSCRIPTIONS_DIR / "asnorm_cohort.npy"
    if _cohort_path.exists():
        voiceprint_db.load_cohort(str(_cohort_path))
        logger.info("AS-norm cohort loaded from %s", _cohort_path)
    else:
        _n = voiceprint_db.build_cohort_from_transcriptions(
            str(TRANSCRIPTIONS_DIR), save_path=str(_cohort_path)
        )
        logger.info("AS-norm cohort built: %d embeddings", _n)
except Exception as _exc:
    logger.warning(
        "AS-norm cohort init failed (identify will use raw cosine): %s", _exc
    )

# In-memory job status
jobs: dict[str, dict] = {}

# Serialise GPU work: only one transcription runs at a time.
# Concurrent HTTP uploads are fine; they queue here before touching the GPU.
import threading as _threading

_gpu_sem = _threading.Semaphore(1)


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
    ffmpeg_timeout = int(os.getenv("FFMPEG_TIMEOUT_SEC", "1800"))
    try:
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
            timeout=ffmpeg_timeout,
        )
    except subprocess.TimeoutExpired:
        wav_path.unlink(missing_ok=True)
        logger.error(
            "ffmpeg timed out after %ds on %s", ffmpeg_timeout, input_path.name
        )
        raise HTTPException(504, f"ffmpeg timed out after {ffmpeg_timeout}s")
    return wav_path


_HASH_INDEX_FILE = TRANSCRIPTIONS_DIR / "hash_index.json"
_hash_index_lock = __import__("threading").Lock()


def _compute_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            sha256.update(chunk)
    return sha256.hexdigest()


def _lookup_hash(file_hash: str) -> str | None:
    """Return existing tr_id if hash is already transcribed and result exists."""
    with _hash_index_lock:
        if not _HASH_INDEX_FILE.exists():
            return None
        index = json.loads(_HASH_INDEX_FILE.read_text())
    tr_id = index.get(file_hash)
    if tr_id and (TRANSCRIPTIONS_DIR / tr_id / "result.json").exists():
        return tr_id
    return None


def _register_hash(file_hash: str, tr_id: str) -> None:
    with _hash_index_lock:
        index = (
            json.loads(_HASH_INDEX_FILE.read_text())
            if _HASH_INDEX_FILE.exists()
            else {}
        )
        index[file_hash] = tr_id
        _HASH_INDEX_FILE.write_text(json.dumps(index, indent=2))


def _run_transcription(
    job_id: str,
    audio_path: Path,
    language: str,
    min_speakers: int,
    max_speakers: int,
    denoise_model: str = None,
    snr_threshold: float = None,
    file_hash: str = None,
):
    """Background transcription worker."""
    try:
        jobs[job_id]["status"] = "converting"
        wav_path = _convert_to_wav(audio_path)

        jobs[job_id]["status"] = "queued"
        with _gpu_sem:
            jobs[job_id]["status"] = (
                "denoising"
                if (denoise_model or DENOISE_MODEL) != "none"
                else "transcribing"
            )
            clean_path = _maybe_denoise(wav_path, denoise_model, snr_threshold)

            # DF peaks at ~15 GB reserved in PyTorch's CUDA cache.
            # ctranslate2 (Whisper) calls cudaMalloc directly and sees the OS
            # free memory — not PyTorch's allocator pool — so it OOMs unless we
            # explicitly flush the cache before Whisper cold-loads.
            try:
                import torch as _torch
                import gc as _gc

                _gc.collect()
                if _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
            except Exception:
                pass

            jobs[job_id]["status"] = "transcribing"
            result = pipeline.process(
                str(clean_path),
                raw_audio_path=str(wav_path),
                language=language,
                min_speakers=min_speakers or None,
                max_speakers=max_speakers or None,
            )

        # Release cached CUDA memory so the next queued job has headroom
        try:
            import torch as _torch
            import gc as _gc

            _gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()
        except Exception:
            pass

        # Match speakers against voiceprint DB
        jobs[job_id]["status"] = "identifying"
        speaker_map = {}
        for spk_label, embedding in result["speaker_embeddings"].items():
            spk_id, spk_name, sim = voiceprint_db.identify(
                embedding, threshold=VOICEPRINT_THRESHOLD
            )
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
        effective_denoise = (denoise_model or DENOISE_MODEL).strip().lower()
        effective_snr = (
            snr_threshold if snr_threshold is not None else DENOISE_SNR_THRESHOLD
        )
        tr = {
            "id": job_id,
            "filename": audio_path.name,
            "created_at": datetime.now().isoformat(),
            "status": "completed",
            "language": language,
            "segments": segments,
            "speaker_map": speaker_map,
            "unique_speakers": result["unique_speakers"],
            "params": {
                "language": language or "auto",
                "denoise_model": effective_denoise,
                "snr_threshold": effective_snr,
                "voiceprint_threshold": VOICEPRINT_THRESHOLD,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            },
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

        if file_hash:
            _register_hash(file_hash, job_id)

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
    language: str = Form(None),
    min_speakers: int = Form(0),
    max_speakers: int = Form(0),
    denoise_model: str = Form("none"),
    snr_threshold: float = Form(None),
):
    # Normalise empty string to None so pipeline treats it as auto-detect.
    language = language.strip() if language else None

    job_id = f"tr_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"

    safe_filename = PurePosixPath(file.filename or "upload").name or "upload"
    save_path = UPLOADS_DIR / f"{job_id}_{safe_filename}"

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

    # Dedup: if identical audio was already transcribed, return existing result.
    file_hash = _compute_file_hash(save_path)
    existing_id = _lookup_hash(file_hash)
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
    thread = Thread(
        target=_run_transcription,
        args=(
            job_id,
            save_path,
            language,
            min_speakers,
            max_speakers,
            denoise_model,
            snr_threshold,
            file_hash,
        ),
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


@app.post("/api/voiceprints/rebuild-cohort")
async def rebuild_cohort():
    """Rebuild the AS-norm cohort from all processed transcriptions."""
    cohort_path = TRANSCRIPTIONS_DIR / "asnorm_cohort.npy"
    n = voiceprint_db.build_cohort_from_transcriptions(
        str(TRANSCRIPTIONS_DIR), save_path=str(cohort_path)
    )
    return {"cohort_size": n, "saved_to": str(cohort_path)}


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
