"""Audio processing utilities.

Covers format conversion (ffmpeg), content-addressed deduplication via a
SHA-256 hash index, and optional noise reduction (DeepFilterNet / noisereduce).
"""

import fcntl
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path

from fastapi import HTTPException

from config import (
    DENOISE_MODEL,
    DENOISE_SNR_THRESHOLD,
    FFMPEG_TIMEOUT_SEC,
    TRANSCRIPTIONS_DIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Input-validation helpers (SEC-C2 / BP-C2)
# ---------------------------------------------------------------------------

_TR_ID_RE = re.compile(r"^tr_[A-Za-z0-9_-]{1,64}$")
_SPEAKER_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CTRL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")


def safe_log_filename(name: str | None) -> str:
    """Strip control chars (incl. CR/LF, ANSI escapes) from user-supplied names
    before writing them to logs, so attackers can't inject fake log lines.
    """
    if not name:
        return ""
    return _CTRL_CHAR_RE.sub("?", name)


def safe_tr_dir(tr_id: str) -> Path:
    """Validate tr_id and return the transcription directory path.

    Raises HTTPException(400) if tr_id contains path traversal characters.
    """
    if not _TR_ID_RE.match(tr_id):
        raise HTTPException(400, f"Invalid transcription ID format: {tr_id!r}")
    path = (TRANSCRIPTIONS_DIR / tr_id).resolve()
    if not str(path).startswith(str(TRANSCRIPTIONS_DIR.resolve())):
        raise HTTPException(400, "Path traversal detected")
    return path


def safe_speaker_label(label: str) -> str:
    """Validate speaker_label to prevent path traversal via filename injection."""
    if not _SPEAKER_LABEL_RE.match(label):
        raise HTTPException(400, f"Invalid speaker label: {label!r}")
    return label


# ---------------------------------------------------------------------------
# ffmpeg conversion
# ---------------------------------------------------------------------------


def convert_to_wav(input_path: Path) -> Path:
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
            timeout=FFMPEG_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired:
        wav_path.unlink(missing_ok=True)
        logger.error(
            "ffmpeg timed out after %ds on %s", FFMPEG_TIMEOUT_SEC, input_path.name
        )
        raise HTTPException(504, f"ffmpeg timed out after {FFMPEG_TIMEOUT_SEC}s")
    return wav_path


# ---------------------------------------------------------------------------
# Content-addressed hash index
# ---------------------------------------------------------------------------

_HASH_INDEX_FILE = TRANSCRIPTIONS_DIR / "hash_index.json"
# CQ-H5: threading.Lock only works within a single process. Replace with an
# fcntl-based file lock so multiple uvicorn workers can safely share the index.
_hash_index_thread_lock = threading.Lock()  # intra-process guard (belt)


def _atomic_write_json(path: Path, data: dict, **json_kwargs) -> None:
    """Write *data* as JSON to *path* atomically via a temp-file rename.
    The rename is POSIX-atomic: readers always see a complete file."""
    content = json.dumps(data, **json_kwargs)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=path.parent,
            delete=False,
            suffix=".tmp",
            encoding="utf-8",
        ) as f:
            tmp_path = f.name
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _with_file_lock(path: Path, func):
    """Execute *func* while holding an exclusive fcntl lock on *path*.lock.

    Falls back to the in-process threading lock on platforms without fcntl
    (e.g. Windows). The thread lock is always acquired first so that two
    threads in the same process don't race through the fcntl acquire.
    """
    lock_path = str(path) + ".lock"
    with _hash_index_thread_lock:
        try:
            with open(lock_path, "w") as lock_f:
                try:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
                    return func()
                finally:
                    fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
        except (AttributeError, OSError):
            # fcntl unavailable (Windows) or lock file can't be opened — the
            # thread lock we already hold is sufficient for single-process use.
            return func()


def compute_file_hash(path: Path) -> str:
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            sha256.update(chunk)
    return sha256.hexdigest()


async def save_upload_and_hash(
    file, save_path: Path, max_bytes: int, chunk_size: int
) -> tuple[int, str]:
    """Save an uploaded UploadFile to *save_path* using async I/O and compute
    its SHA-256 digest on the fly, avoiding a second full-file read.

    Returns (total_bytes_written, hex_digest).

    Raises ValueError when the upload exceeds *max_bytes* — the caller is
    responsible for unlinking *save_path* and returning HTTP 413.

    PERF-C2: replaces the former synchronous open()+write() loop and the
    separate compute_file_hash() call, both of which blocked the asyncio
    event loop for several seconds on large files.
    """
    import aiofiles

    sha256 = hashlib.sha256()
    size = 0
    async with aiofiles.open(save_path, "wb") as f:
        while True:
            chunk = await file.read(chunk_size)
            if not chunk:
                break
            size += len(chunk)
            if size > max_bytes:
                raise ValueError(f"Upload exceeds MAX_UPLOAD_BYTES ({max_bytes} bytes)")
            await f.write(chunk)
            sha256.update(chunk)
    return size, sha256.hexdigest()


def lookup_hash(file_hash: str) -> str | None:
    """Return existing tr_id if hash is already transcribed and result exists."""

    def _do():
        if not _HASH_INDEX_FILE.exists():
            return None
        return json.loads(_HASH_INDEX_FILE.read_text()).get(file_hash)

    tr_id = _with_file_lock(_HASH_INDEX_FILE, _do)
    if tr_id and (TRANSCRIPTIONS_DIR / tr_id / "result.json").exists():
        return tr_id
    return None


def register_hash(file_hash: str, tr_id: str) -> None:
    def _do():
        index = (
            json.loads(_HASH_INDEX_FILE.read_text())
            if _HASH_INDEX_FILE.exists()
            else {}
        )
        index[file_hash] = tr_id
        _atomic_write_json(_HASH_INDEX_FILE, index, indent=2)

    _with_file_lock(_HASH_INDEX_FILE, _do)


# ---------------------------------------------------------------------------
# DeepFilterNet / noise reduction
# ---------------------------------------------------------------------------

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


def maybe_denoise(
    wav_path: Path, model: str = None, snr_threshold: float = None
) -> Path:
    """Return denoised WAV path if DENOISE_MODEL is set; otherwise return wav_path unchanged."""
    effective_model = (model or DENOISE_MODEL).strip().lower()
    if effective_model == "none":
        return wav_path

    threshold = snr_threshold if snr_threshold is not None else DENOISE_SNR_THRESHOLD
    out_path = wav_path.with_suffix(".denoised.wav")

    if effective_model == "deepfilternet":
        import torch
        import torchaudio

        snr_db = _estimate_snr(wav_path)
        if snr_db >= threshold:
            logger.info("DeepFilterNet skipped (SNR=%.1fdB, clean audio)", snr_db)
            return wav_path

        logger.info(
            "DeepFilterNet applying (SNR=%.1fdB < %.1fdB threshold)",
            snr_db,
            threshold,
        )
        df_model, df_state = _load_deepfilternet()
        import df as _df_pkg

        audio, sr = torchaudio.load(str(wav_path))
        if sr != df_state.sr():
            audio = torchaudio.functional.resample(audio, sr, df_state.sr())
        audio = audio.contiguous()
        with torch.backends.cudnn.flags(enabled=False):
            enhanced = _df_pkg.enhance(df_model, df_state, audio)
        torchaudio.save(
            str(out_path),
            enhanced.unsqueeze(0) if enhanced.dim() == 1 else enhanced,
            df_state.sr(),
        )
        logger.info("DeepFilterNet: denoised %s → %s", wav_path.name, out_path.name)

    elif effective_model == "noisereduce":
        import soundfile as sf
        import noisereduce as nr

        data, sr = sf.read(str(wav_path), dtype="float32")
        reduced = nr.reduce_noise(y=data, sr=sr, stationary=True)
        sf.write(str(out_path), reduced, sr)
        logger.info("noisereduce: denoised %s → %s", wav_path.name, out_path.name)

    else:
        logger.warning("Unknown DENOISE_MODEL=%r — skipping denoising", effective_model)
        return wav_path

    return out_path
