"""TEST-C1: core security paths.

Covers:
- API-key auth: rejects when key missing, accepts both Bearer and X-API-Key
- Path traversal: tr_id regex rejects malicious transcription IDs
- SEC-C1 regression: enroll_speaker loads .npy with allow_pickle=False so a
  pickle-laden .npy cannot execute arbitrary code
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

_APP_DIR = Path(__file__).resolve().parent.parent / "app"
_REPO_ROOT = Path(__file__).resolve().parent.parent


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
            if _m in ("main", "config") or _m.startswith(("api.", "infra.")) or _m in {
                "api",
                "infra",
            }:
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


def _auth_headers() -> dict[str, str]:
    return {
        "Authorization": "Bearer s3cret",
        "X-API-Key": "s3cret",
    }


def test_pyannote_checkpoint_loading_uses_scoped_safe_globals_only():
    """PyTorch checkpoint allowlists must stay scoped and weights-only safe."""

    source = (_REPO_ROOT / "app" / "pipeline" / "orchestrator.py").read_text(
        encoding="utf-8"
    )

    assert "safe_globals(" in source
    assert "add_safe_globals" not in source
    assert not re.search(r"weights_only\s*=\s*False", source)


def _seed_result_json(
    tmpdir: Path,
    tr_id: str,
    *,
    filename: str = "tr_demo.wav",
    payload: dict | None = None,
    raw_text: str | None = None,
) -> Path:
    tr_dir = tmpdir / "transcriptions" / tr_id
    tr_dir.mkdir(parents=True, exist_ok=True)
    result_path = tr_dir / "result.json"
    if raw_text is not None:
        result_path.write_text(raw_text, encoding="utf-8")
        return result_path

    data = payload or {
        "id": tr_id,
        "filename": filename,
        "created_at": "2026-04-23T00:00:00+00:00",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 1.0,
                "speaker_label": "SPEAKER_00",
                "speaker_name": "Alice",
                "text": "hello",
            }
        ],
        "unique_speakers": ["Alice"],
        "speaker_map": {},
    }
    result_path.write_text(json.dumps(data), encoding="utf-8")
    return result_path


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


def test_public_allowlist_paths_bypass_auth_only_for_exact_entries(monkeypatch):
    """Public docs/UI routes must stay reachable, but lookalikes stay protected."""
    with _AuthedClientCtx("s3cret", monkeypatch) as (client, _tmpdir):
        for path in [
            "/",
            "/healthz",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/static/index.html",
        ]:
            resp = client.get(path)
            assert resp.status_code != 401, f"{path} should stay public: {resp.text}"

        for path in [
            "/docsXYZ",
            "/redocx",
            "/openapi.json/extra",
            "/static",
            "/static..",
        ]:
            resp = client.get(path)
            assert resp.status_code == 401, (
                f"{path} must not inherit public allowlist access: "
                f"{resp.status_code} {resp.text}"
            )


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


def test_transcribe_sanitizes_filename_and_inflight_deduplicates(monkeypatch):
    """A control-char filename must be sanitized and duplicate bytes must reuse the first job."""
    with _AuthedClientCtx("s3cret", monkeypatch) as (client, tmpdir):
        import api.routers.transcriptions as transcriptions

        class _FakeThread:
            def __init__(self, *args, **kwargs):
                self.args = args
                self.kwargs = kwargs

            def start(self):
                return None

        monkeypatch.setattr(transcriptions, "Thread", _FakeThread)

        files = {"file": ("../-y\nattack.wav", b"same-bytes", "audio/wav")}
        first = client.post("/api/transcribe", files=files, headers=_auth_headers())
        assert first.status_code == 200, first.text
        body = first.json()
        job_id = body["id"]
        assert body["status"] == "queued"

        sanitized = transcriptions.jobs[job_id]["filename"]
        assert sanitized.startswith("-y")
        assert ".." not in sanitized
        assert "\n" not in sanitized and "\r" not in sanitized
        upload_files = list((tmpdir / "uploads").iterdir())
        assert len(upload_files) == 1
        assert upload_files[0].name == f"{job_id}_{sanitized}"
        assert "\n" not in upload_files[0].name and "\r" not in upload_files[0].name

        second = client.post("/api/transcribe", files=files, headers=_auth_headers())
        assert second.status_code == 200, second.text
        assert second.json() == {
            "id": job_id,
            "status": "queued",
            "deduplicated": True,
        }
        assert len(list((tmpdir / "uploads").iterdir())) == 1


def test_oversized_upload_returns_413_and_cleans_partial_file(monkeypatch):
    """Oversized uploads must fail early and leave no partial artifact behind."""
    monkeypatch.setenv("MAX_UPLOAD_BYTES", "8")
    with _AuthedClientCtx("s3cret", monkeypatch) as (client, tmpdir):
        resp = client.post(
            "/api/transcribe",
            files={"file": ("large.wav", b"0123456789ABCDEF", "audio/wav")},
            headers=_auth_headers(),
        )

        assert resp.status_code == 413, resp.text
        assert list((tmpdir / "uploads").iterdir()) == []


def test_corrupt_status_json_never_500s(monkeypatch):
    """Corrupt status.json should be treated as missing job, not a 500."""
    with _AuthedClientCtx("s3cret", monkeypatch) as (client, tmpdir):
        tr_id = "tr_badstatus"
        tr_dir = tmpdir / "transcriptions" / tr_id
        tr_dir.mkdir(parents=True, exist_ok=True)
        (tr_dir / "status.json").write_text("{not-json", encoding="utf-8")

        resp = client.get(f"/api/jobs/{tr_id}", headers=_auth_headers())
        assert resp.status_code == 404, resp.text


@pytest.mark.parametrize(
    ("method", "path", "kwargs"),
    [
        ("get", "/api/transcriptions/tr_corrupt", {}),
        ("get", "/api/transcriptions/tr_corrupt/audio", {}),
        ("get", "/api/export/tr_corrupt?format=txt", {}),
        (
            "put",
            "/api/transcriptions/tr_corrupt/segments/0/speaker",
            {"data": {"speaker_name": "Alice"}},
        ),
    ],
)
def test_corrupt_result_json_returns_controlled_error(monkeypatch, method, path, kwargs):
    """Corrupt result.json must not crash read/edit/export endpoints."""
    with _AuthedClientCtx("s3cret", monkeypatch) as (client, tmpdir):
        _seed_result_json(tmpdir, "tr_corrupt", raw_text="{definitely-not-json")
        (tmpdir / "uploads" / "tr_demo.wav").write_bytes(b"audio")

        resp = getattr(client, method)(path, headers=_auth_headers(), **kwargs)
        assert resp.status_code == 409, (
            f"{path} should surface a controlled corrupt-artifact error, "
            f"got {resp.status_code}: {resp.text}"
        )


def test_export_txt_sanitizes_speaker_name_control_chars(monkeypatch):
    """speaker_name must not inject extra lines into TXT export output."""
    with _AuthedClientCtx("s3cret", monkeypatch) as (client, tmpdir):
        _seed_result_json(
            tmpdir,
            "tr_export_inject",
            payload={
                "id": "tr_export_inject",
                "filename": "tr_demo.wav",
                "created_at": "2026-04-23T00:00:00+00:00",
                "segments": [
                    {
                        "id": 0,
                        "start": 0.0,
                        "end": 1.0,
                        "speaker_label": "SPEAKER_00",
                        "speaker_name": "Alice\r\nInjected: yes",
                        "text": "hello",
                    }
                ],
                "unique_speakers": ["Alice\r\nInjected: yes"],
                "speaker_map": {},
            },
        )

        resp = client.get(
            "/api/export/tr_export_inject?format=txt",
            headers=_auth_headers(),
        )

        assert resp.status_code == 200, resp.text
        lines = resp.text.splitlines()
        assert len(lines) == 1, (
            f"speaker_name control chars must not create extra export lines: {lines!r}"
        )
        assert "\r" not in resp.text
        assert "Alice Injected: yes" in lines[0]
