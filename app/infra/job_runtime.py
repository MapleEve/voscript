"""Infrastructure helpers for in-memory job runtime state."""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import TypeVar

from config import JOBS_MAX_CACHE

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

# In-flight dedup: prevents two concurrent requests with identical audio from
# both burning GPU. Cleared when the job reaches a terminal state.
_in_flight_hashes: dict[str, str] = {}
_in_flight_lock = threading.Lock()


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
        result = work()
    flush_torch_cuda_cache(logger, phase="post-pipeline")
    return result


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
    "register_in_flight",
    "run_serialized_gpu_work",
    "unregister_in_flight",
]
