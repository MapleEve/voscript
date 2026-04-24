"""Centralised configuration — all os.getenv() calls live here.

Modules import from this file rather than calling os.getenv() directly so
that environment-variable names are defined in exactly one place and are
easy to audit.
"""

import os
from pathlib import Path


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_csv_set(name: str, default: str = "") -> frozenset[str]:
    raw = os.getenv(name)
    if raw is None:
        raw = default
    return frozenset(item.strip().lower() for item in raw.split(",") if item.strip())


def _env_mapping(name: str) -> dict[str, str]:
    raw = os.getenv(name, "").strip()
    mapping: dict[str, str] = {}
    if not raw:
        return mapping
    for item in raw.split(","):
        key, separator, value = item.partition("=")
        if separator and key.strip() and value.strip():
            mapping[key.strip().lower()] = value.strip()
    return mapping


# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

DATA_DIR: Path = Path(os.getenv("DATA_DIR", "/data"))
TRANSCRIPTIONS_DIR: Path = DATA_DIR / "transcriptions"
UPLOADS_DIR: Path = DATA_DIR / "uploads"
VOICEPRINTS_DIR: Path = DATA_DIR / "voiceprints"
MODELS_DIR: Path = Path(os.getenv("MODELS_DIR", "/models"))

# ---------------------------------------------------------------------------
# Auth / CORS
# ---------------------------------------------------------------------------

API_KEY: str | None = (os.getenv("API_KEY") or "").strip() or None

# SEC-C3: allow operators to explicitly acknowledge running without auth.
# When ALLOW_NO_AUTH=1 the warning is suppressed; the service still runs open.
ALLOW_NO_AUTH: bool = os.getenv("ALLOW_NO_AUTH", "0") == "1"

CORS_ORIGINS: str = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()

# ---------------------------------------------------------------------------
# Upload limits
# ---------------------------------------------------------------------------

# Cap how much any single upload can occupy on disk. Whisper + pyannote
# comfortably handle 2 GB of audio (~20 h @ typical bitrates); anything
# beyond that is either a mistake or an attempt to exhaust storage.
MAX_UPLOAD_BYTES: int = int(os.getenv("MAX_UPLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
UPLOAD_CHUNK: int = 1 << 20  # 1 MiB

# ---------------------------------------------------------------------------
# Model / inference settings
# ---------------------------------------------------------------------------

WHISPER_MODEL: str = os.getenv("WHISPER_MODEL", "large-v3")
HF_TOKEN: str | None = os.getenv("HF_TOKEN")
DEVICE: str = os.getenv("DEVICE", "cuda")
LANGUAGE: str = os.getenv("LANGUAGE", "")

# WhisperX forced-alignment controls. Languages are attempted by default; use
# WHISPERX_ALIGN_DISABLED_LANGUAGES only for an explicit operational fallback.
WHISPERX_ALIGN_DISABLED_LANGUAGES: frozenset[str] = _env_csv_set(
    "WHISPERX_ALIGN_DISABLED_LANGUAGES",
    "",
)
WHISPERX_ALIGN_MODEL_MAP: dict[str, str] = _env_mapping("WHISPERX_ALIGN_MODEL_MAP")
WHISPERX_ALIGN_MODEL_DIR: str | None = (
    os.getenv("WHISPERX_ALIGN_MODEL_DIR", "").strip() or None
)
WHISPERX_ALIGN_CACHE_ONLY: bool = os.getenv("WHISPERX_ALIGN_CACHE_ONLY", "0") == "1"

# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

DENOISE_MODEL: str = os.getenv("DENOISE_MODEL", "none").strip().lower()

# SNR threshold (dB) below which DeepFilterNet is applied.
# Audio estimated at or above this level is considered clean and skipped,
# matching the A/B finding that DF hurts high-quality recordings (e.g. PLAUD Pin).
DENOISE_SNR_THRESHOLD: float = _env_float("DENOISE_SNR_THRESHOLD", 10.0)

# ---------------------------------------------------------------------------
# Speaker identification
# ---------------------------------------------------------------------------

# Base cosine-similarity threshold for voiceprint identify(). The actual
# threshold per candidate is adaptive — see voiceprint_db.identify's docstring
# for the per-speaker relaxation rules.
VOICEPRINT_THRESHOLD: float = _env_float("VOICEPRINT_THRESHOLD", 0.75)

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

FFMPEG_TIMEOUT_SEC: int = int(os.getenv("FFMPEG_TIMEOUT_SEC", "1800"))
JOBS_MAX_CACHE: int = int(os.getenv("JOBS_MAX_CACHE", "200"))

# Paths that must stay open even when API_KEY auth is enabled. "/" is the
# bundled web UI (browsers can't attach a Bearer header to a direct
# navigation — the UI's own fetch() calls to /api/* still carry the key).
# /static/* serves the UI's assets. /healthz is a liveness probe. /docs
# /redoc /openapi.json are FastAPI's auto docs.
PUBLIC_EXACT_PATHS: frozenset = frozenset(
    {
        "/",
        "/healthz",
        "/docs",
        "/redoc",
        "/openapi.json",
    }
)
PUBLIC_PATH_PREFIXES: tuple = ("/static/",)
