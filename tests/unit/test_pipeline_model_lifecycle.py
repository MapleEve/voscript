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


def _install_fake_faster_whisper(monkeypatch, loaded_models):
    class FakeWhisperModel:
        def __init__(self, model_ref, **kwargs):
            loaded_models.append((model_ref, kwargs))

    faster_whisper = ModuleType("faster_whisper")
    faster_whisper.WhisperModel = FakeWhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", faster_whisper)
    return FakeWhisperModel


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
    loaded_models = []
    fake_model = _install_fake_faster_whisper(monkeypatch, loaded_models)

    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        orchestrator,
        "select_best_cuda_device",
        lambda configured: calls.append(configured) or "cuda:1",
    )

    assert pipeline.whisper.__class__ is fake_model

    assert calls == ["cuda"]
    assert loaded_models == [
        ("tiny", {"device": "cuda", "device_index": 1, "compute_type": "float16"})
    ]
    assert pipeline.device == "cuda:1"


def test_cpu_lazy_load_does_not_probe_cuda(monkeypatch):
    pipeline = _new_pipeline(device="cpu")
    loaded_models = []
    fake_model = _install_fake_faster_whisper(monkeypatch, loaded_models)

    def fail_if_called(configured):
        raise AssertionError("CPU-only loads must not probe CUDA")

    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(orchestrator, "select_best_cuda_device", fail_if_called)

    assert pipeline.whisper.__class__ is fake_model

    assert loaded_models == [("tiny", {"device": "cpu", "compute_type": "int8"})]
    assert pipeline.device == "cpu"


def test_whisper_lazy_load_keeps_unindexed_cuda_supported(monkeypatch):
    pipeline = _new_pipeline(device="cuda")
    loaded_models = []
    fake_model = _install_fake_faster_whisper(monkeypatch, loaded_models)

    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        orchestrator, "select_best_cuda_device", lambda configured: configured
    )

    assert pipeline.whisper.__class__ is fake_model

    assert loaded_models == [("tiny", {"device": "cuda", "compute_type": "float16"})]
    assert pipeline.device == "cuda"


def test_whisper_lazy_load_normalizes_cuda_zero_for_faster_whisper(monkeypatch):
    pipeline = _new_pipeline(device="cuda:0")
    loaded_models = []
    fake_model = _install_fake_faster_whisper(monkeypatch, loaded_models)

    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        orchestrator, "select_best_cuda_device", lambda configured: configured
    )

    assert pipeline.whisper.__class__ is fake_model

    assert loaded_models == [
        ("tiny", {"device": "cuda", "device_index": 0, "compute_type": "float16"})
    ]
    assert pipeline.device == "cuda:0"


def test_whisper_lazy_load_normalizes_fallback_cuda_index_for_faster_whisper(
    monkeypatch,
):
    pipeline = _new_pipeline(device="cuda:1")
    loaded_models = []
    fake_model = _install_fake_faster_whisper(monkeypatch, loaded_models)

    monkeypatch.setattr(orchestrator.Path, "exists", lambda self: False)
    monkeypatch.setattr(
        orchestrator, "select_best_cuda_device", lambda configured: configured
    )

    assert pipeline.whisper.__class__ is fake_model

    assert loaded_models == [
        ("tiny", {"device": "cuda", "device_index": 1, "compute_type": "float16"})
    ]
    assert pipeline.device == "cuda:1"
