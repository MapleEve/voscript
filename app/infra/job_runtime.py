"""Infrastructure helpers for in-memory job runtime state."""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from config import JOBS_MAX_CACHE, MODEL_IDLE_TIMEOUT_SEC

_MISSING = object()
_T = TypeVar("_T")


class _LRUJobsDict:
    """Thread-safe LRU dict for job states with bounded size."""

    def __init__(self, maxsize: int = 200):
        self._d: OrderedDict = OrderedDict()
        self._lock = threading.Lock()
        self._maxsize = maxsize

    def __setitem__(self, key, value):
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
            self._d[key] = value
            if len(self._d) > self._maxsize:
                self._d.popitem(last=False)

    def __getitem__(self, key):
        with self._lock:
            return self._d[key]

    def __contains__(self, key):
        with self._lock:
            return key in self._d

    def __delitem__(self, key):
        with self._lock:
            del self._d[key]

    def get(self, key, default=None):
        with self._lock:
            return self._d.get(key, default)

    def pop(self, key, default=_MISSING):
        with self._lock:
            if default is _MISSING:
                return self._d.pop(key)
            return self._d.pop(key, default)


jobs: _LRUJobsDict = _LRUJobsDict(maxsize=JOBS_MAX_CACHE)

# Serialise GPU work: only one transcription runs at a time.
# Concurrent HTTP uploads are fine; they queue here before touching the GPU.
_gpu_sem = threading.Semaphore(1)
_runtime_state_lock = threading.Lock()
_last_gpu_job_finished_at: float | None = None

# In-flight dedup: prevents two concurrent requests with identical audio from
# both burning GPU. Cleared when the job reaches a terminal state.
_in_flight_hashes: dict[str, str] = {}
_in_flight_lock = threading.Lock()


@dataclass(frozen=True)
class IdleModelUnloadDaemon:
    thread: threading.Thread
    stop_event: threading.Event

    def stop(self, timeout: float = 5.0) -> None:
        self.stop_event.set()
        self.thread.join(timeout=timeout)


def flush_torch_cuda_cache(
    logger: logging.Logger | None = None,
    *,
    phase: str,
) -> None:
    """Best-effort CUDA cache flush used around serialized GPU work."""

    try:
        import gc as _gc

        import torch as _torch

        _gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - guarded for runtime-only failures
        if logger is not None:
            logger.warning("%s CUDA cache flush failed: %s", phase, exc)


def run_serialized_gpu_work(
    work: Callable[[], _T],
    *,
    logger: logging.Logger | None = None,
) -> _T:
    """Execute GPU work under the shared semaphore with current cache policy."""

    with _gpu_sem:
        flush_torch_cuda_cache(logger, phase="pre-whisper")
        try:
            result = work()
        finally:
            record_gpu_job_finished()
    flush_torch_cuda_cache(logger, phase="post-pipeline")
    return result


def record_gpu_job_finished(*, finished_at: float | None = None) -> None:
    """Record the latest completed GPU job time for idle-unload decisions."""

    global _last_gpu_job_finished_at
    with _runtime_state_lock:
        _last_gpu_job_finished_at = (
            time.monotonic() if finished_at is None else finished_at
        )


def _last_finished_at() -> float | None:
    with _runtime_state_lock:
        return _last_gpu_job_finished_at


def _model_is_loaded(pipeline) -> bool:
    has_loaded_models = getattr(pipeline, "has_loaded_models", None)
    if has_loaded_models is None:
        return False
    return bool(has_loaded_models())


def _is_idle_due(timeout_s: float, *, now: Callable[[], float]) -> bool:
    finished_at = _last_finished_at()
    if finished_at is None:
        return False
    return now() - finished_at >= timeout_s


def unload_idle_pipeline_if_due(
    pipeline,
    *,
    timeout_s: float = MODEL_IDLE_TIMEOUT_SEC,
    now: Callable[[], float] = time.monotonic,
    logger: logging.Logger | None = None,
) -> bool:
    """Unload loaded models when the serialized GPU runtime has been idle.

    The idle state is checked before waiting and again after acquiring the GPU
    semaphore. The second check is what prevents unloading based on a stale
    observation made before a newer queued job completed.
    """

    if timeout_s <= 0 or not _model_is_loaded(pipeline):
        return False
    if not _is_idle_due(timeout_s, now=now):
        return False

    _gpu_sem.acquire()
    try:
        if not _model_is_loaded(pipeline):
            return False
        if not _is_idle_due(timeout_s, now=now):
            return False
        pipeline.unload_models()
        flush_torch_cuda_cache(logger, phase="idle-unload")
        return True
    finally:
        _gpu_sem.release()


def _idle_model_unload_worker(
    pipeline,
    *,
    timeout_s: float,
    interval_s: float,
    stop_event: threading.Event,
    logger: logging.Logger | None,
) -> None:
    while not stop_event.wait(interval_s):
        try:
            unload_idle_pipeline_if_due(
                pipeline,
                timeout_s=timeout_s,
                logger=logger,
            )
        except Exception:
            if logger is not None:
                logger.exception("idle model unload worker tick failed")


def start_idle_model_unload_daemon(
    pipeline,
    *,
    timeout_s: float = MODEL_IDLE_TIMEOUT_SEC,
    interval_s: float | None = None,
    logger: logging.Logger | None = None,
) -> IdleModelUnloadDaemon | None:
    """Start the optional idle-unload daemon, or return None when disabled."""

    if timeout_s <= 0:
        return None

    stop_event = threading.Event()
    tick_interval = (
        interval_s if interval_s is not None else min(max(timeout_s / 2, 1.0), 60.0)
    )
    thread = threading.Thread(
        target=_idle_model_unload_worker,
        kwargs={
            "pipeline": pipeline,
            "timeout_s": timeout_s,
            "interval_s": tick_interval,
            "stop_event": stop_event,
            "logger": logger,
        },
        daemon=True,
        name="idle-model-unload",
    )
    thread.start()
    return IdleModelUnloadDaemon(thread=thread, stop_event=stop_event)


def register_in_flight(file_hash: str, job_id: str) -> str | None:
    """Register hash as in-flight.

    Returns existing job_id if a concurrent job is already processing the same
    content, None if registered successfully.
    """
    with _in_flight_lock:
        if file_hash in _in_flight_hashes:
            return _in_flight_hashes[file_hash]
        _in_flight_hashes[file_hash] = job_id
        return None


def unregister_in_flight(file_hash: str, job_id: str | None = None) -> bool:
    with _in_flight_lock:
        current_job = _in_flight_hashes.get(file_hash)
        if current_job is None:
            return False
        if job_id is not None and current_job != job_id:
            return False
        _in_flight_hashes.pop(file_hash, None)
        return True


__all__ = [
    "_LRUJobsDict",
    "flush_torch_cuda_cache",
    "jobs",
    "record_gpu_job_finished",
    "register_in_flight",
    "run_serialized_gpu_work",
    "start_idle_model_unload_daemon",
    "unload_idle_pipeline_if_due",
    "unregister_in_flight",
]
