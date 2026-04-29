"""Hugging Face model loading helpers with cache-first safeguards."""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def configure_huggingface_runtime() -> None:
    """Set safe Hugging Face Hub defaults before importing hub clients.

    ``huggingface_hub`` reads environment flags at import time.  Set defaults
    here so deployments avoid the Xet/CAS download path unless operators opt in.
    """

    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "3")


def resolve_cached_hf_snapshot(repo_id: str, token: str | None = None) -> str | None:
    """Return a cached snapshot path for *repo_id* without network access."""

    configure_huggingface_runtime()
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            repo_id=repo_id,
            token=token,
            local_files_only=True,
        )
    except Exception:
        logger.info(
            "No complete local Hugging Face cache for model; using configured Hub download path."
        )
        return None


def resolve_hf_model_ref(
    repo_id: str,
    *,
    token: str | None = None,
    purpose: str,
) -> str:
    """Return a local cached snapshot when available, otherwise the repo id."""

    cached_snapshot = resolve_cached_hf_snapshot(repo_id, token=token)
    if cached_snapshot:
        logger.info("Loading %s from local Hugging Face cache.", purpose)
        return cached_snapshot

    logger.info(
        "Loading %s from Hugging Face Hub with Xet disabled by default.",
        purpose,
    )
    return repo_id


configure_huggingface_runtime()


__all__ = [
    "configure_huggingface_runtime",
    "resolve_hf_model_ref",
    "resolve_cached_hf_snapshot",
]
