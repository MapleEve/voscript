"""Infrastructure helpers for persisted job status and recovery."""

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from config import TRANSCRIPTIONS_DIR

logger = logging.getLogger(__name__)


def _atomic_write_json(path: Path, payload: dict, **json_kwargs) -> None:
    """Write JSON atomically: write to temp file in same dir, then os.replace()."""
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", dir=parent, delete=False, suffix=".tmp", encoding="utf-8"
        ) as tf:
            tmp_path = tf.name
            json.dump(payload, tf, **json_kwargs)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _write_status(
    job_id: str,
    status: str,
    error: str | None = None,
    filename: str | None = None,
) -> bool:
    """Write job status to disk for persistence across process restarts.

    Returns True on success, False on failure. Callers that need durability
    (e.g. the initial "queued" write before the worker thread starts) should
    check the return value and abort if False.
    """
    status_path = TRANSCRIPTIONS_DIR / job_id / "status.json"
    try:
        payload = {
            "status": status,
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            "error": error,
        }
        if filename is not None:
            payload["filename"] = filename
        _atomic_write_json(status_path, payload)
        return True
    except Exception as exc:
        logger.warning("Failed to write status.json for %s: %s", job_id, exc)
        return False


def recover_orphan_jobs() -> None:
    """Mark any in-progress jobs as failed if the process was restarted."""
    try:
        for status_path in TRANSCRIPTIONS_DIR.glob("*/status.json"):
            try:
                data = json.loads(status_path.read_text())
                if data.get("status") not in ("completed", "failed"):
                    data["status"] = "failed"
                    data["error"] = "Process restarted while job was in progress"
                    data["updated_at"] = datetime.now(tz=timezone.utc).isoformat()
                    _atomic_write_json(status_path, data)
                    logger.info(
                        "AR-C2: marked orphan job %s as failed",
                        status_path.parent.name,
                    )
            except Exception as exc:
                logger.warning(
                    "AR-C2: could not recover orphan job at %s: %s", status_path, exc
                )
    except Exception as exc:
        logger.warning("AR-C2: orphan job recovery scan failed: %s", exc)
