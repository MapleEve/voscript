"""Unit tests for CUDA device selection helpers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace


class _FakeCuda:
    def __init__(self, *, available=True, free_memory=None, fail=False):
        self.available = available
        self.free_memory = free_memory or []
        self.fail = fail

    def is_available(self):
        return self.available

    def device_count(self):
        return len(self.free_memory)

    def mem_get_info(self, index):
        if self.fail:
            raise RuntimeError("probe failed")
        return self.free_memory[index], 100


def test_select_best_cuda_device_uses_most_free_memory():
    select_best_cuda_device = importlib.import_module(
        "infra.cuda_devices"
    ).select_best_cuda_device
    torch_module = SimpleNamespace(
        cuda=_FakeCuda(available=True, free_memory=[400, 900, 800])
    )

    assert select_best_cuda_device("cuda", torch_module=torch_module) == "cuda:1"


def test_select_best_cuda_device_falls_back_to_configured_device_on_probe_failure():
    select_best_cuda_device = importlib.import_module(
        "infra.cuda_devices"
    ).select_best_cuda_device
    torch_module = SimpleNamespace(
        cuda=_FakeCuda(available=True, free_memory=[400, 900], fail=True)
    )

    assert select_best_cuda_device("cuda:0", torch_module=torch_module) == "cuda:0"


def test_select_best_cuda_device_keeps_cpu_and_no_cuda_unchanged():
    select_best_cuda_device = importlib.import_module(
        "infra.cuda_devices"
    ).select_best_cuda_device
    cpu_torch = SimpleNamespace(cuda=_FakeCuda(available=True, free_memory=[900]))
    no_cuda_torch = SimpleNamespace(cuda=_FakeCuda(available=False, free_memory=[900]))

    assert select_best_cuda_device("cpu", torch_module=cpu_torch) == "cpu"
    assert select_best_cuda_device("cuda", torch_module=no_cuda_torch) == "cuda"
