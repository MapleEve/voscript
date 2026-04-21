"""FastAPI dependency callables shared across all routers."""

import hmac

from fastapi import Header, HTTPException, Request

from config import API_KEY


async def verify_api_key(x_api_key: str | None = Header(None)) -> None:
    """Dependency that enforces API-key authentication.

    Raises HTTP 403 when a key is configured but the supplied value does not
    match.  When no key is configured (open mode) the dependency is a no-op.

    Note: the middleware in main.py handles the Bearer-token path and path
    allow-listing; this dependency is the fallback for router-level auth.
    """
    if API_KEY is None:
        return  # open mode — no check needed
    if not x_api_key or not hmac.compare_digest(x_api_key, API_KEY):
        raise HTTPException(403, "Invalid API key")


def get_db(request: Request):
    """Return the VoiceprintDB instance stored on app.state."""
    return request.app.state.db


def get_pipeline(request: Request):
    """Return the TranscriptionPipeline instance stored on app.state."""
    return request.app.state.pipeline
