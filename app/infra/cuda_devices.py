"""CUDA device selection helpers."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def select_best_cuda_device(
    configured_device: str,
    *,
    torch_module: Any | None = None,
) -> str:
    """Return the visible CUDA device with the most free memory.

    Probe failures are non-fatal: callers keep the configured device rather
    than failing a lazy model load on telemetry-only logic.
    """

    if not configured_device.startswith("cuda"):
        return configured_device

    try:
        if torch_module is None:
            import torch as torch_module

        cuda = torch_module.cuda
        if not cuda.is_available():
            return configured_device

        device_count = cuda.device_count()
        if device_count <= 0:
            return configured_device

        free_by_device = [
            (cuda.mem_get_info(index)[0], index) for index in range(device_count)
        ]
        _, best_index = max(free_by_device)
        return f"cuda:{best_index}"
    except Exception as exc:  # pragma: no cover - depends on host CUDA runtime
        logger.warning(
            "CUDA free-memory probe failed; keeping %s: %s", configured_device, exc
        )
        return configured_device
