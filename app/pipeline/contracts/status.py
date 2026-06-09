"""Stable contract helpers for persisted job status payloads."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import PurePosixPath
import re
from typing import Any

JOB_STATUS_QUEUED = "queued"
JOB_STATUS_CONVERTING = "converting"
JOB_STATUS_DENOISING = "denoising"
JOB_STATUS_TRANSCRIBING = "transcribing"
JOB_STATUS_IDENTIFYING = "identifying"
JOB_STATUS_COMPLETED = "completed"
JOB_STATUS_FAILED = "failed"

IN_PROGRESS_JOB_STATUSES = frozenset(
    {
        JOB_STATUS_QUEUED,
        JOB_STATUS_CONVERTING,
        JOB_STATUS_DENOISING,
        JOB_STATUS_TRANSCRIBING,
        JOB_STATUS_IDENTIFYING,
    }
)
TERMINAL_JOB_STATUSES = frozenset({JOB_STATUS_COMPLETED, JOB_STATUS_FAILED})
KNOWN_JOB_STATUSES = IN_PROGRESS_JOB_STATUSES | TERMINAL_JOB_STATUSES

_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _public_safe_text(value: Any) -> str:
    return _CONTROL_RE.sub(" ", str(value or "")).strip()


def _public_safe_filename(value: Any) -> str | None:
    if value is None:
        return None
    text = _public_safe_text(value).replace("\\", "/")
    filename = PurePosixPath(text).name
    return filename or None


def normalize_job_status(status: Any, *, default: str = JOB_STATUS_FAILED) -> str:
    """Return a known status, defaulting invalid legacy values to failed."""

    value = _public_safe_text(status).lower()
    if value in KNOWN_JOB_STATUSES:
        return value
    return default


def build_status_payload(
    status: str,
    *,
    error: Any = None,
    filename: Any = None,
    updated_at: str | None = None,
) -> dict[str, Any]:
    """Build the persisted status.json payload without changing API shape."""

    normalized_status = normalize_job_status(status, default="")
    if not normalized_status:
        raise ValueError(f"unknown job status: {status!r}")

    payload: dict[str, Any] = {
        "status": normalized_status,
        "updated_at": updated_at or _utc_now_iso(),
        "error": None if error is None else _public_safe_text(error),
    }
    safe_filename = _public_safe_filename(filename)
    if safe_filename is not None:
        payload["filename"] = safe_filename
    return payload


def normalize_status_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Normalize legacy status.json payloads for restart/recovery reads."""

    if not isinstance(payload, Mapping):
        return build_status_payload(
            JOB_STATUS_FAILED,
            error="Invalid persisted job status",
        )

    updated_at = payload.get("updated_at")
    if not isinstance(updated_at, str) or not updated_at.strip():
        updated_at = _utc_now_iso()

    return build_status_payload(
        normalize_job_status(payload.get("status")),
        error=payload.get("error"),
        filename=payload.get("filename"),
        updated_at=updated_at,
    )


__all__ = [
    "IN_PROGRESS_JOB_STATUSES",
    "JOB_STATUS_COMPLETED",
    "JOB_STATUS_CONVERTING",
    "JOB_STATUS_DENOISING",
    "JOB_STATUS_FAILED",
    "JOB_STATUS_IDENTIFYING",
    "JOB_STATUS_QUEUED",
    "JOB_STATUS_TRANSCRIBING",
    "KNOWN_JOB_STATUSES",
    "TERMINAL_JOB_STATUSES",
    "build_status_payload",
    "normalize_job_status",
    "normalize_status_payload",
]
