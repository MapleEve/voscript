"""Unit tests for pipeline model unload and reload-time device selection."""

from __future__ import annotations

import sys
from types import ModuleType

if "numpy" not in sys.modules:
    numpy_stub = ModuleType("numpy")
    numpy_stub.ndarray = object
    sys.modules["numpy"] = numpy_stub

from pipeline import TranscriptionPipeline
import pipeline.orchestrator as orchestrator


def _new_pipeline(*, device="cuda"):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = device
    pipeline._configured_device = device
    pipeline.model_size = "tiny"
    pipeline.hf_token = None
    pipeline._whisper = None
    pipeline._diarization = None
    pipeline._embedding_model = None
    pipeline._runner = None
    return pipeline


def test_unload_models_drops_loaded_references_without_selecting_device(monkeypatch):
    pipeline = _new_pipeline(device="cuda")
    pipeline._whisper = object()
    pipeline._diarization = object()
    pipeline._embedding_model = object()
    calls = []

    monkeypatch.setattr(
        orchestrator,
        "select_best_cuda_device",
        lambda configured: calls.append(configured) or "cuda:1",
    )

    assert pipeline.has_loaded_models() is True

    pipeline.unload_models()

    assert pipeline.has_loaded_models() is False
    assert pipeline._whisper is None
    assert pipeline._diarization is None
    assert pipeline._embedding_model is None
    assert calls == []


def test_whisper_lazy_reload_selects_best_cuda_device(monkeypatch):
    pipeline = _new_pipeline(device="cuda")
    calls = []
    loaded_devices = []

    class FakeWhisperModel:
        def __init__(self, model_ref, *, device, compute_type):
            loaded_devices.append((model_ref, device, compute_type))

    faster_whisper = ModuleType("faster_whisper")
    faster_whisper.WhisperModel = FakeWhisperModel

    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", faster_whisper)
    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        orchestrator,
        "select_best_cuda_device",
        lambda configured: calls.append(configured) or "cuda:1",
    )

    assert pipeline.whisper.__class__ is FakeWhisperModel

    assert calls == ["cuda"]
    assert loaded_devices == [("tiny", "cuda:1", "float16")]
    assert pipeline.device == "cuda:1"


def test_cpu_lazy_load_does_not_probe_cuda(monkeypatch):
    pipeline = _new_pipeline(device="cpu")
    loaded_devices = []

    class FakeWhisperModel:
        def __init__(self, model_ref, *, device, compute_type):
            loaded_devices.append((device, compute_type))

    faster_whisper = ModuleType("faster_whisper")
    faster_whisper.WhisperModel = FakeWhisperModel

    def fail_if_called(configured):
        raise AssertionError("CPU-only loads must not probe CUDA")

    monkeypatch.setitem(__import__("sys").modules, "faster_whisper", faster_whisper)
    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(orchestrator, "select_best_cuda_device", fail_if_called)

    assert pipeline.whisper.__class__ is FakeWhisperModel

    assert loaded_devices == [("cpu", "int8")]
    assert pipeline.device == "cpu"
