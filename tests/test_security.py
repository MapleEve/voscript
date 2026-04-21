"""TEST-C1: core security paths.

Covers:
- API-key auth: rejects when key missing, accepts both Bearer and X-API-Key
- Path traversal: tr_id regex rejects malicious transcription IDs
- SEC-C1 regression: enroll_speaker loads .npy with allow_pickle=False so a
  pickle-laden .npy cannot execute arbitrary code
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "app"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _AuthedClientCtx:
    """Context manager that boots a TestClient with a configured API_KEY.

    Using a context manager (rather than a bare factory) guarantees FastAPI's
    ``lifespan`` runs — which is where TRANSCRIPTIONS_DIR et al. are created.
    Forgetting to enter the lifespan leaves those directories missing and
    every listing endpoint immediately 500s on iterdir().
    """

    def __init__(self, api_key: str, monkeypatch):
        self.api_key = api_key
        self.monkeypatch = monkeypatch
        self.tmpdir = Path(tempfile.mkdtemp(prefix="voscript-test-sec-"))

    def __enter__(self):
        from fastapi.testclient import TestClient

        self.monkeypatch.setenv("DATA_DIR", str(self.tmpdir))
        self.monkeypatch.setenv("API_KEY", self.api_key)
        self.monkeypatch.chdir(_APP_DIR)

        for _m in list(sys.modules):
            if _m in ("main", "config") or _m.startswith("api.") or _m == "api":
                del sys.modules[_m]
            elif _m.startswith("services."):
                del sys.modules[_m]

        from main import app  # noqa: WPS433 — late import on purpose

        # raise_server_exceptions=False so 500-level errors come back as
        # HTTP responses rather than bubbling through as Python exceptions —
        # the SEC-C1 test must be able to inspect a 500 from np.load().
        self._client_cm = TestClient(app, raise_server_exceptions=False)
        return self._client_cm.__enter__(), self.tmpdir

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._client_cm.__exit__(exc_type, exc, tb)
        finally:
            import shutil

            shutil.rmtree(self.tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------


def test_api_key_required_when_set(monkeypatch):
    """Without any credentials the middleware must reject with 401."""
    with _AuthedClientCtx("s3cret", monkeypatch) as (client, _tmpdir):
        resp = client.get("/api/transcriptions")
        assert (
            resp.status_code == 401
        ), f"expected 401 Unauthorized, got {resp.status_code}: {resp.text}"
        assert resp.headers.get("www-authenticate", "").lower().startswith("bearer")


def test_api_key_accepted_via_bearer(monkeypatch):
    """Authorization: Bearer <key> must be accepted by the auth middleware.

    The app has two layers: the ``require_api_key`` middleware in main.py
    accepts **either** Bearer or X-API-Key, and the router-level
    ``verify_api_key`` dependency is a belt-and-braces fallback that only
    inspects X-API-Key. We verify the middleware's Bearer branch by
    observing that a correct Bearer does NOT produce 401 (the middleware's
    terminal response), and that sending both headers completes the round
    trip with a 200.
    """
    with _AuthedClientCtx("s3cret", monkeypatch) as (client, _tmpdir):
        # Wrong Bearer → middleware 401s.
        resp = client.get(
            "/api/transcriptions",
            headers={"Authorization": "Bearer WRONG"},
        )
        assert (
            resp.status_code == 401
        ), f"middleware must 401 on bad Bearer, got {resp.status_code}"

        # Correct Bearer → middleware no longer 401s (it may 403 at the
        # router-level dep, but that's a distinct gate, not a middleware
        # rejection).
        resp = client.get(
            "/api/transcriptions",
            headers={"Authorization": "Bearer s3cret"},
        )
        assert resp.status_code != 401, (
            f"Bearer auth was rejected by middleware: "
            f"{resp.status_code} {resp.text}"
        )

        # End-to-end: Bearer + X-API-Key together must yield 200.
        resp = client.get(
            "/api/transcriptions",
            headers={
                "Authorization": "Bearer s3cret",
                "X-API-Key": "s3cret",
            },
        )
        assert (
            resp.status_code == 200
        ), f"Bearer + X-API-Key must pass: {resp.status_code} {resp.text}"
        assert resp.json() == []


def test_api_key_accepted_via_x_api_key_header(monkeypatch):
    """X-API-Key header should also pass the middleware (constant-time compare)."""
    with _AuthedClientCtx("s3cret", monkeypatch) as (client, _tmpdir):
        resp = client.get(
            "/api/transcriptions",
            headers={"X-API-Key": "s3cret"},
        )
        assert resp.status_code == 200
        assert resp.json() == []


# ---------------------------------------------------------------------------
# Path-traversal defence (SEC-C2)
# ---------------------------------------------------------------------------


def test_path_traversal_rejected(monkeypatch):
    """tr_id that contains .. must be refused before any filesystem access.

    Both auth headers are sent so the request clears the two auth layers and
    we actually exercise the validation regex (otherwise a 403 from the
    router-level dep would mask the path-traversal check and hide drift).
    """
    with _AuthedClientCtx("s3cret", monkeypatch) as (client, _tmpdir):
        headers = {
            "Authorization": "Bearer s3cret",
            "X-API-Key": "s3cret",
        }

        # Percent-encoded traversal: FastAPI's path regex rejects at the
        # routing layer (422). safe_tr_dir also raises 400 for anything else.
        resp = client.get("/api/transcriptions/..%2F..%2Fetc%2Fpasswd", headers=headers)
        assert resp.status_code in (
            400,
            404,
            422,
        ), f"path traversal must be refused, got {resp.status_code}: {resp.text}"

        # The regex ``^tr_[A-Za-z0-9_-]{1,64}$`` disallows '.' entirely, so a
        # tr_id with embedded .. components must be rejected too (defence in
        # depth, independent of any filesystem resolution).
        resp = client.get("/api/transcriptions/tr_..etc", headers=headers)
        assert resp.status_code in (
            400,
            404,
            422,
        ), f"tr_id with '.' must be refused, got {resp.status_code}: {resp.text}"

        # The same endpoint with a legitimate (nonexistent) id must 404, not
        # 400/422 — this confirms the rejections above are due to the regex,
        # not a stray auth failure.
        resp = client.get("/api/transcriptions/tr_does_not_exist", headers=headers)
        assert (
            resp.status_code == 404
        ), f"well-formed unknown id should 404, got {resp.status_code}"


# ---------------------------------------------------------------------------
# SEC-C1: np.load(allow_pickle=False) on enroll embedding path
# ---------------------------------------------------------------------------


def _build_pickle_npy(path: Path) -> None:
    """Write a .npy that, when loaded with ``allow_pickle=True``, would unpickle
    a Python object. We use a harmless dict so the file is safe even if the
    load accidentally succeeded.
    """
    payload = np.array([{"executed": True}], dtype=object)
    np.save(str(path), payload, allow_pickle=True)


def test_np_load_allow_pickle_false(monkeypatch):
    """enroll_speaker must refuse to load an object-dtype .npy (pickle payload).

    Pins the SEC-C1 defence: ``np.load(emb_path, allow_pickle=False)`` — if a
    future refactor drops the flag, this test flips red instantly.
    """
    with _AuthedClientCtx("s3cret", monkeypatch) as (client, tmpdir):
        tr_id = "tr_pickle_probe"
        tr_dir = tmpdir / "transcriptions" / tr_id
        tr_dir.mkdir(parents=True, exist_ok=True)
        (tr_dir / "result.json").write_text("{}")

        emb_path = tr_dir / "emb_SPEAKER_00.npy"
        _build_pickle_npy(emb_path)

        resp = client.post(
            "/api/voiceprints/enroll",
            data={
                "tr_id": tr_id,
                "speaker_label": "SPEAKER_00",
                "speaker_name": "pickle-probe",
            },
            headers={
                "Authorization": "Bearer s3cret",
                "X-API-Key": "s3cret",
            },
        )

        # Auth is clean (both headers sent) → we must have reached the
        # loader. np.load(..., allow_pickle=False) raises ValueError, and
        # FastAPI surfaces it as 500. 401/403 here would mean we tested the
        # auth stack, not the SEC-C1 defence — so we reject those explicitly.
        assert resp.status_code not in (401, 403), (
            f"auth unexpectedly failed; cannot assert SEC-C1 "
            f"(status={resp.status_code}, body={resp.text!r})"
        )
        assert resp.status_code >= 400, (
            f"SEC-C1 regression: enroll_speaker accepted pickle-laden .npy "
            f"(status={resp.status_code}, body={resp.text!r})"
        )

        # Belt-and-braces: confirm the defence at the numpy layer itself.
        with pytest.raises(ValueError, match="(?i)allow_pickle"):
            np.load(str(emb_path), allow_pickle=False)
