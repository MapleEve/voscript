"""Unit tests for GPU runtime serialization helpers."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

import infra.job_runtime as job_runtime


@pytest.fixture(autouse=True)
def _reset_idle_runtime_state(monkeypatch):
    monkeypatch.setattr(job_runtime, "_last_gpu_job_finished_at", None, raising=False)


class _FakePipeline:
    def __init__(self, *, loaded: bool = True):
        self.loaded = loaded
        self.unload_calls = 0

    def has_loaded_models(self) -> bool:
        return self.loaded

    def unload_models(self) -> None:
        self.unload_calls += 1
        self.loaded = False


def test_run_serialized_gpu_work_flushes_before_and_after_success(monkeypatch):
    events = []

    monkeypatch.setattr(
        job_runtime,
        "flush_torch_cuda_cache",
        lambda logger=None, *, phase: events.append(phase),
    )

    result = job_runtime.run_serialized_gpu_work(lambda: events.append("work") or "ok")

    assert result == "ok"
    assert events == ["pre-whisper", "work", "post-pipeline"]


def test_run_serialized_gpu_work_skips_post_flush_on_error(monkeypatch):
    events = []

    monkeypatch.setattr(
        job_runtime,
        "flush_torch_cuda_cache",
        lambda logger=None, *, phase: events.append(phase),
    )

    with pytest.raises(RuntimeError, match="boom"):
        job_runtime.run_serialized_gpu_work(
            lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

    assert events == ["pre-whisper"]


def test_run_serialized_gpu_work_releases_semaphore_after_error(monkeypatch):
    events = []

    monkeypatch.setattr(
        job_runtime,
        "flush_torch_cuda_cache",
        lambda logger=None, *, phase: events.append(phase),
    )

    with pytest.raises(RuntimeError, match="boom"):
        job_runtime.run_serialized_gpu_work(
            lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        )

    result = job_runtime.run_serialized_gpu_work(lambda: events.append("retry") or "ok")

    assert result == "ok"
    assert events == ["pre-whisper", "pre-whisper", "retry", "post-pipeline"]


def test_runtime_admission_count_helpers(monkeypatch):
    cache = job_runtime._LRUJobsDict(maxsize=10)
    cache["queued"] = {"status": "queued"}
    cache["converting"] = {"status": "converting"}
    cache["done"] = {"status": "completed"}
    cache["failed"] = {"status": "failed"}
    monkeypatch.setattr(job_runtime, "jobs", cache)
    monkeypatch.setattr(job_runtime, "_active_job_ids", {"tr_queued", "tr_converting"})
    monkeypatch.setattr(
        job_runtime,
        "_in_flight_hashes",
        {"sha256:a": "tr_a", "sha256:b": "tr_b"},
    )

    assert job_runtime.active_job_count() == 2
    assert job_runtime.in_flight_count() == 2


def test_runtime_job_store_public_api_tracks_current_store(monkeypatch):
    cache = job_runtime._LRUJobsDict(maxsize=2)
    monkeypatch.setattr(job_runtime, "jobs", cache)

    job_runtime.set_runtime_job("tr_public", {"status": "queued"})

    assert job_runtime.runtime_job_exists("tr_public") is True
    assert job_runtime.get_runtime_job("tr_public") == {"status": "queued"}
    assert job_runtime.runtime_job_count() == 1
    assert job_runtime.runtime_jobs_values_snapshot() == ({"status": "queued"},)

    updated = job_runtime.update_runtime_job(
        "tr_public",
        {"status": "completed", "result": {"id": "tr_public"}},
    )

    assert updated == {
        "status": "completed",
        "result": {"id": "tr_public"},
    }
    assert job_runtime.get_runtime_job("missing", {"status": "missing"}) == {
        "status": "missing",
    }
    assert job_runtime.pop_runtime_job("tr_public") == {
        "status": "completed",
        "result": {"id": "tr_public"},
    }
    assert job_runtime.runtime_job_exists("tr_public") is False
    assert job_runtime.pop_runtime_job("missing", None) is None


def test_active_job_reservation_is_not_coupled_to_lru_eviction(monkeypatch):
    monkeypatch.setattr(job_runtime, "_active_job_ids", set())
    cache = job_runtime._LRUJobsDict(maxsize=1)
    monkeypatch.setattr(job_runtime, "jobs", cache)

    reserved = job_runtime.try_reserve_active_job("tr_old", max_entries=1)
    cache["tr_old"] = {"status": "queued"}
    cache["tr_new"] = {"status": "queued"}
    rejected = job_runtime.try_reserve_active_job("tr_new", max_entries=1)

    assert reserved.reserved is True
    assert "tr_old" not in cache
    assert rejected.budget_exceeded is True
    assert job_runtime.active_job_count() == 1
    assert job_runtime.release_active_job("tr_old") is True
    assert job_runtime.active_job_count() == 0


def test_try_register_in_flight_enforces_budget_atomically(monkeypatch):
    monkeypatch.setattr(job_runtime, "_in_flight_hashes", {"sha256:a": "tr_a"})

    duplicate = job_runtime.try_register_in_flight(
        "sha256:a",
        "tr_duplicate",
        max_entries=1,
    )
    rejected = job_runtime.try_register_in_flight(
        "sha256:b",
        "tr_b",
        max_entries=1,
    )
    admitted = job_runtime.try_register_in_flight(
        "sha256:c",
        "tr_c",
        max_entries=0,
    )

    assert duplicate.existing_job_id == "tr_a"
    assert duplicate.registered is False
    assert duplicate.budget_exceeded is False
    assert rejected.existing_job_id is None
    assert rejected.registered is False
    assert rejected.budget_exceeded is True
    assert admitted.registered is True


def test_flush_torch_cuda_cache_skips_python_gc_for_active_job_phases(monkeypatch):
    events = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: events.append("empty_cache"),
        )
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        job_runtime,
        "_collect_python_gc",
        lambda: events.append("gc_collect"),
    )

    job_runtime.flush_torch_cuda_cache(phase="post-pipeline")

    assert events == ["empty_cache"]


def test_flush_torch_cuda_cache_keeps_full_gc_for_idle_unload(monkeypatch):
    events = []
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: events.append("empty_cache"),
        )
    )

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(
        job_runtime,
        "_collect_python_gc",
        lambda: events.append("gc_collect"),
    )

    job_runtime.flush_torch_cuda_cache(phase="idle-unload")

    assert events == ["gc_collect", "empty_cache"]


def test_idle_unload_daemon_disabled_when_timeout_zero():
    pipeline = _FakePipeline(loaded=True)

    daemon = job_runtime.start_idle_model_unload_daemon(
        pipeline,
        timeout_s=0,
        interval_s=0.01,
    )
    unloaded = job_runtime.unload_idle_pipeline_if_due(
        pipeline,
        timeout_s=0,
        now=lambda: 100.0,
    )

    assert daemon is None
    assert unloaded is False
    assert pipeline.unload_calls == 0


def test_idle_unload_runs_after_loaded_model_exceeds_timeout(monkeypatch):
    pipeline = _FakePipeline(loaded=True)
    events = []

    monkeypatch.setattr(
        job_runtime,
        "flush_torch_cuda_cache",
        lambda logger=None, *, phase: events.append(phase),
    )

    job_runtime.record_gpu_job_finished(finished_at=10.0)

    unloaded = job_runtime.unload_idle_pipeline_if_due(
        pipeline,
        timeout_s=5,
        now=lambda: 16.0,
    )

    assert unloaded is True
    assert pipeline.loaded is False
    assert pipeline.unload_calls == 1
    assert events == ["idle-unload"]


def test_idle_unload_skips_when_no_model_is_loaded():
    pipeline = _FakePipeline(loaded=False)
    job_runtime.record_gpu_job_finished(finished_at=10.0)

    unloaded = job_runtime.unload_idle_pipeline_if_due(
        pipeline,
        timeout_s=5,
        now=lambda: 16.0,
    )

    assert unloaded is False
    assert pipeline.unload_calls == 0


def test_idle_unload_rechecks_idle_after_waiting_for_gpu_semaphore(monkeypatch):
    pipeline = _FakePipeline(loaded=True)
    job_runtime.record_gpu_job_finished(finished_at=10.0)

    class UpdatingSemaphore:
        def __init__(self):
            self.released = False

        def acquire(self):
            # Simulate the daemon waiting while a newer job finishes. The
            # post-acquire idle check must see this fresher completion time.
            job_runtime.record_gpu_job_finished(finished_at=98.0)
            return True

        def release(self):
            self.released = True

    semaphore = UpdatingSemaphore()
    monkeypatch.setattr(job_runtime, "_gpu_sem", semaphore)

    unloaded = job_runtime.unload_idle_pipeline_if_due(
        pipeline,
        timeout_s=5,
        now=lambda: 100.0,
    )

    assert unloaded is False
    assert pipeline.unload_calls == 0
    assert semaphore.released is True
