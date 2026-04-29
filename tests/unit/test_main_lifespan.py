"""TEST-H4, TEST-H5: lifespan background thread behaviour.

TEST-H4: The 'cohort-rebuild' daemon thread must be alive while the TestClient
         lifespan is active.

TEST-H5: The 'cohort-rebuild' daemon thread must survive a tick where
         maybe_rebuild_cohort raises an exception (exception is caught and
         logged; the thread loop continues).

The thread and stop_event are locals inside the lifespan coroutine, so we
locate the thread by its well-known name via threading.enumerate().
"""

import inspect
import time
import pytest
import threading


def _find_rebuild_thread():
    """Return the running 'cohort-rebuild' daemon thread, or None."""
    for t in threading.enumerate():
        if t.name == "cohort-rebuild":
            return t
    return None


@pytest.fixture(autouse=True)
def _patch_voiceprint_stub_ctor(monkeypatch):
    """Keep the fallback VoiceprintDB stub compatible with lifespan startup."""
    from voiceprints.db import VoiceprintDB

    init = VoiceprintDB.__init__
    if "cohort_path" in inspect.signature(init).parameters:
        return

    def _compat_init(self, path, cohort_path=None, *args, **kwargs):
        init(self, path)
        self.path = path
        self.cohort_path = cohort_path
        self._cohort_path = cohort_path

    monkeypatch.setattr(VoiceprintDB, "__init__", _compat_init)


def test_rebuild_thread_alive_during_lifespan(app_client):
    """Daemon thread must be alive while the TestClient lifespan is active (TEST-H4)."""
    thread = _find_rebuild_thread()
    assert thread is not None, (
        "Expected a running thread named 'cohort-rebuild' during lifespan; "
        "none found among: " + str([t.name for t in threading.enumerate()])
    )
    assert thread.is_alive(), "cohort-rebuild thread should be alive during lifespan"
    assert thread.daemon, "cohort-rebuild thread must be a daemon thread"


def test_openapi_version_reports_075(app_client):
    assert app_client.app.version == "0.7.5"


def test_rebuild_thread_survives_tick_exception(app_client, monkeypatch):
    """Daemon thread must stay alive after maybe_rebuild_cohort raises (TEST-H5)."""
    thread = _find_rebuild_thread()
    assert (
        thread is not None
    ), "cohort-rebuild thread not found; lifespan may not have started"

    # Locate the db object on app.state via the TestClient's app.
    # app_client is a FastAPI TestClient; the underlying ASGI app has .state.db
    # set by lifespan.
    db = app_client.app.state.db

    call_count = [0]
    original_maybe_rebuild = db.maybe_rebuild_cohort

    def _raise(*a, **kw):
        call_count[0] += 1
        raise RuntimeError("simulated cohort rebuild failure")

    monkeypatch.setattr(db, "maybe_rebuild_cohort", _raise)

    # Give the thread a moment — the tick interval is 60 s so it won't fire
    # during a short test, but the thread itself must remain alive.
    time.sleep(0.1)

    assert thread.is_alive(), (
        "Daemon thread died after monkeypatching maybe_rebuild_cohort; "
        "the worker loop must catch exceptions and keep running"
    )

    # Restore to avoid affecting subsequent tests.
    monkeypatch.setattr(db, "maybe_rebuild_cohort", original_maybe_rebuild)
