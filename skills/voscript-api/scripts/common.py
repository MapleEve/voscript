"""VoScript API client - shared configuration and HTTP helpers.

This module provides the ``VoScriptClient`` class that every helper script
in this package uses to talk to a running VoScript FastAPI server.

Auth:
    VoScript accepts either ``X-API-Key: <key>`` header or
    ``Authorization: Bearer <key>``. This client uses ``X-API-Key`` by default.

Environment variables (fallback when constructor args are omitted):
    VOSCRIPT_URL        Base URL of the VoScript server (e.g. http://localhost:7880)
    VOSCRIPT_API_KEY    API key registered on the server

Dependencies: stdlib + ``requests`` only.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Iterable, Optional, Tuple

import requests


DEFAULT_TIMEOUT = 120  # seconds, for regular JSON requests
UPLOAD_TIMEOUT = 3600  # seconds, for multipart uploads (large audio files)


class VoScriptError(RuntimeError):
    """Raised when the VoScript server returns a non-2xx response."""

    def __init__(self, status_code: int, message: str, payload: Any = None) -> None:
        super().__init__(f"[{status_code}] {message}")
        self.status_code = status_code
        self.message = message
        self.payload = payload


class VoScriptClient:
    """VoScript API client - configuration and HTTP helpers.

    Parameters
    ----------
    url:
        Base URL of the VoScript server. Falls back to ``$VOSCRIPT_URL``.
    api_key:
        API key. Falls back to ``$VOSCRIPT_API_KEY``.
    timeout:
        Default request timeout in seconds (used for non-upload requests).
    """

    def __init__(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        resolved_url = url or os.environ.get("VOSCRIPT_URL")
        resolved_key = api_key or os.environ.get("VOSCRIPT_API_KEY")

        if not resolved_url:
            raise ValueError(
                "VoScript server URL not provided. Pass --url or set "
                "VOSCRIPT_URL environment variable."
            )
        if not resolved_key:
            raise ValueError(
                "VoScript API key not provided. Pass --api-key or set "
                "VOSCRIPT_API_KEY environment variable."
            )

        self.url = resolved_url.rstrip("/")
        self.api_key = resolved_key
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "X-API-Key": self.api_key,
                "Accept": "application/json",
            }
        )

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------
    def _build_url(self, path: str) -> str:
        if path.startswith("http://") or path.startswith("https://"):
            return path
        if not path.startswith("/"):
            path = "/" + path
        return self.url + path

    def _parse_response(self, resp: requests.Response) -> Any:
        if resp.status_code >= 400:
            try:
                payload = resp.json()
                message = payload.get("detail") or payload.get("message") or resp.text
            except ValueError:
                payload = None
                message = resp.text or resp.reason
            raise VoScriptError(resp.status_code, str(message), payload)

        if not resp.content:
            return None

        content_type = resp.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                return resp.json()
            except ValueError:
                return resp.text
        return resp.text

    # ------------------------------------------------------------------
    # Public HTTP methods
    # ------------------------------------------------------------------
    def get(
        self, path: str, params: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> Any:
        resp = self._session.get(
            self._build_url(path),
            params=params,
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )
        return self._parse_response(resp)

    def post(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        # Larger timeout when uploading files
        default_timeout = UPLOAD_TIMEOUT if files else self.timeout
        resp = self._session.post(
            self._build_url(path),
            data=data,
            files=files,
            json=json_body,
            timeout=kwargs.pop("timeout", default_timeout),
            **kwargs,
        )
        return self._parse_response(resp)

    def put(
        self,
        path: str,
        data: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        resp = self._session.put(
            self._build_url(path),
            data=data,
            json=json_body,
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )
        return self._parse_response(resp)

    def delete(self, path: str, **kwargs: Any) -> Any:
        resp = self._session.delete(
            self._build_url(path),
            timeout=kwargs.pop("timeout", self.timeout),
            **kwargs,
        )
        return self._parse_response(resp)

    def download(
        self, path: str, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[bytes, str]:
        """Download binary content. Returns ``(bytes, suggested_filename)``."""
        resp = self._session.get(
            self._build_url(path),
            params=params,
            timeout=self.timeout,
            stream=False,
        )
        if resp.status_code >= 400:
            try:
                payload = resp.json()
                message = payload.get("detail") or payload.get("message") or resp.text
            except ValueError:
                payload = None
                message = resp.text or resp.reason
            raise VoScriptError(resp.status_code, str(message), payload)

        filename = _filename_from_content_disposition(
            resp.headers.get("content-disposition", "")
        )
        return resp.content, filename


# ----------------------------------------------------------------------
# CLI helpers reused by every script
# ----------------------------------------------------------------------
def add_common_args(parser: "object") -> None:
    """Attach ``--url`` and ``--api-key`` to an ``argparse.ArgumentParser``.

    Typed as ``object`` to avoid importing argparse at module import time
    (scripts import argparse themselves).
    """
    parser.add_argument(
        "--url",
        default=None,
        help="VoScript server base URL (falls back to $VOSCRIPT_URL).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="VoScript API key (falls back to $VOSCRIPT_API_KEY).",
    )


def build_client_from_args(args: Any) -> VoScriptClient:
    """Build a ``VoScriptClient`` from parsed argparse args."""
    return VoScriptClient(url=args.url, api_key=args.api_key)


def print_json(value: Any) -> None:
    """Print a value as pretty JSON (UTF-8, no ASCII escaping)."""
    print(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=False))


def format_hms(seconds: float) -> str:
    """Format a number of seconds as ``HH:MM:SS.mmm`` (SRT-ish)."""
    if seconds is None:
        return "--:--:--.---"
    total_ms = int(round(float(seconds) * 1000))
    hours, rem = divmod(total_ms, 3600 * 1000)
    minutes, rem = divmod(rem, 60 * 1000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _filename_from_content_disposition(header: str) -> str:
    """Extract filename from ``Content-Disposition`` header, else empty string."""
    if not header:
        return ""
    parts: Iterable[str] = (p.strip() for p in header.split(";"))
    for part in parts:
        if part.lower().startswith("filename*="):
            # RFC 5987: filename*=UTF-8''encoded-name
            value = part.split("=", 1)[1]
            if "''" in value:
                value = value.split("''", 1)[1]
            return value.strip().strip('"')
        if part.lower().startswith("filename="):
            return part.split("=", 1)[1].strip().strip('"')
    return ""


__all__ = [
    "VoScriptClient",
    "VoScriptError",
    "add_common_args",
    "build_client_from_args",
    "print_json",
    "format_hms",
]
