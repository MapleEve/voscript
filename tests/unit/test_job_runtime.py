"""Unit tests for GPU runtime serialization helpers."""

from __future__ import annotations

import pytest

import infra.job_runtime as job_runtime


def test_run_serialized_gpu_work_flushes_before_and_after_success(monkeypatch):
    events = []

    monkeypatch.setattr(
        job_runtime,
        "flush_torch_cuda_cache",
        lambda logger=None, *, phase: events.append(phase),
    )

    result = job_runtime.run_serialized_gpu_work(
        lambda: events.append("work") or "ok"
    )

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
        job_runtime.run_serialized_gpu_work(lambda: (_ for _ in ()).throw(RuntimeError("boom")))

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

    result = job_runtime.run_serialized_gpu_work(
        lambda: events.append("retry") or "ok"
    )

    assert result == "ok"
    assert events == ["pre-whisper", "pre-whisper", "retry", "post-pipeline"]
