"""Filesystem and log-safety helpers for audio-related endpoints."""

from __future__ import annotations

import re
from pathlib import Path

from fastapi import HTTPException

from config import TRANSCRIPTIONS_DIR

_TR_ID_RE = re.compile(r"^tr_[A-Za-z0-9_-]{1,64}$")
_SPEAKER_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_CTRL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")


def safe_log_filename(name: str | None) -> str:
    """Strip control chars so user-supplied filenames cannot forge logs."""

    if not name:
        return ""
    return _CTRL_CHAR_RE.sub("?", name)


def safe_tr_dir(tr_id: str) -> Path:
    """Validate tr_id and return its transcription directory."""

    if not _TR_ID_RE.match(tr_id):
        raise HTTPException(400, f"Invalid transcription ID format: {tr_id!r}")
    path = (TRANSCRIPTIONS_DIR / tr_id).resolve()
    if not str(path).startswith(str(TRANSCRIPTIONS_DIR.resolve())):
        raise HTTPException(400, "Path traversal detected")
    return path


def safe_speaker_label(label: str) -> str:
    """Validate speaker labels before embedding them into filenames."""

    if not _SPEAKER_LABEL_RE.match(label):
        raise HTTPException(400, f"Invalid speaker label: {label!r}")
    return label


__all__ = ["safe_log_filename", "safe_speaker_label", "safe_tr_dir"]
