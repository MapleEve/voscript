"""FastAPI service for voice transcription with speaker identification."""

import hmac
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routers import health, transcriptions, voiceprints
from config import (
    ALLOW_NO_AUTH,
    API_KEY,
    CORS_ORIGINS,
    HF_TOKEN,
    PUBLIC_EXACT_PATHS,
    PUBLIC_PATH_PREFIXES,
    TRANSCRIPTIONS_DIR,
    UPLOADS_DIR,
    VOICEPRINTS_DIR,
    WHISPER_MODEL,
    DEVICE,
)
from infra.job_persistence import recover_orphan_jobs
from pipeline import TranscriptionPipeline
from voiceprints.db import VoiceprintDB

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: startup / teardown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ensure data directories exist
    for d in [TRANSCRIPTIONS_DIR, UPLOADS_DIR, VOICEPRINTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    cohort_path = TRANSCRIPTIONS_DIR / "asnorm_cohort.npy"

    # AR-C2: mark any in-progress jobs from a previous process as failed so
    # frontend polls receive a definitive terminal state on restart.
    recover_orphan_jobs()

    # Initialise voiceprint DB and AS-norm cohort
    db = VoiceprintDB(str(VOICEPRINTS_DIR), cohort_path=str(cohort_path))
    try:
        if cohort_path.exists():
            db.load_cohort(str(cohort_path))
            logger.info("AS-norm cohort loaded from %s", cohort_path)
        else:
            _n = db.build_cohort_from_transcriptions(str(TRANSCRIPTIONS_DIR))
            logger.info("AS-norm cohort built: %d embeddings", _n)
    except Exception as exc:
        logger.warning(
            "AS-norm cohort init failed (identify will use raw cosine): %s", exc
        )
    app.state.db = db

    # Background daemon: auto-rebuild AS-norm cohort after new enrollments.
    _stop_event = threading.Event()

    def _cohort_rebuild_worker(
        db, transcriptions_dir, stop_event, interval_s=60, debounce_s=30
    ):
        while not stop_event.wait(timeout=interval_s):
            try:
                db.maybe_rebuild_cohort(str(transcriptions_dir), debounce_s=debounce_s)
            except Exception:
                logger.exception("cohort-rebuild worker tick failed")

    _rebuild_thread = threading.Thread(
        target=_cohort_rebuild_worker,
        args=(db, TRANSCRIPTIONS_DIR, _stop_event),
        daemon=True,
        name="cohort-rebuild",
    )
    _rebuild_thread.start()

    # Initialise transcription pipeline
    app.state.pipeline = TranscriptionPipeline(WHISPER_MODEL, DEVICE, HF_TOKEN)

    # Auth mode warning
    if API_KEY is None and not ALLOW_NO_AUTH:
        logger.warning(
            "API_KEY is not set. Service is OPEN to all requests. "
            "Set API_KEY env var or set ALLOW_NO_AUTH=1 to suppress this warning."
        )
    elif API_KEY is None:
        logger.warning(
            "API_KEY is not set and ALLOW_NO_AUTH=1. "
            "The service is accepting unauthenticated requests intentionally."
        )
    else:
        logger.info("API_KEY auth enabled for /api/* and / (Bearer or X-API-Key).")

    yield

    _stop_event.set()
    _rebuild_thread.join(timeout=5)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(title="VoScript", version="0.7.3", lifespan=lifespan)

# CORS
_cors_origins = [o.strip() for o in CORS_ORIGINS.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "X-API-Key",
        "X-Request-Id",
    ],
    expose_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-XSS-Protection", "1; mode=block")
    return response


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


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router)
app.include_router(transcriptions.router)
app.include_router(voiceprints.router)
