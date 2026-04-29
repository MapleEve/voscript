"""Unit tests for GPU runtime serialization helpers."""

from __future__ import annotations

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
