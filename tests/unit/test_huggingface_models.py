"""Unit tests for Hugging Face model loading safeguards."""

from __future__ import annotations

import sys
import os
from types import ModuleType


def _stub_numpy(monkeypatch) -> None:
    numpy_stub = ModuleType("numpy")
    numpy_stub.ndarray = object
    monkeypatch.setitem(sys.modules, "numpy", numpy_stub)


def test_configure_huggingface_runtime_disables_xet_by_default(monkeypatch):
    monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)
    monkeypatch.delenv("HF_HUB_ETAG_TIMEOUT", raising=False)
    monkeypatch.delitem(sys.modules, "huggingface_hub", raising=False)
    monkeypatch.delitem(sys.modules, "hf_xet", raising=False)

    from infra.huggingface_models import configure_huggingface_runtime

    configure_huggingface_runtime()

    assert "huggingface_hub" not in sys.modules
    assert "hf_xet" not in sys.modules
    assert os.environ["HF_HUB_DISABLE_XET"] == "1"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "3"


def test_configure_huggingface_runtime_preserves_explicit_operator_values(
    monkeypatch,
):
    monkeypatch.setenv("HF_HUB_DISABLE_XET", "0")
    monkeypatch.setenv("HF_HUB_ETAG_TIMEOUT", "12")

    from infra.huggingface_models import configure_huggingface_runtime

    configure_huggingface_runtime()

    assert os.environ["HF_HUB_DISABLE_XET"] == "0"
    assert os.environ["HF_HUB_ETAG_TIMEOUT"] == "12"


def test_resolve_cached_hf_snapshot_prefers_existing_cache(monkeypatch, tmp_path):
    cached_snapshot = tmp_path / "models--pyannote--speaker-diarization-3.1"
    calls = []

    fake_hub = ModuleType("huggingface_hub")

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(cached_snapshot)

    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    from infra.huggingface_models import resolve_cached_hf_snapshot

    resolved = resolve_cached_hf_snapshot(
        "pyannote/speaker-diarization-3.1",
        token="secret-token",
    )

    assert resolved == str(cached_snapshot)
    assert calls == [
        {
            "repo_id": "pyannote/speaker-diarization-3.1",
            "token": "secret-token",
            "local_files_only": True,
        }
    ]


def test_resolve_cached_hf_snapshot_returns_none_when_cache_missing(monkeypatch):
    fake_hub = ModuleType("huggingface_hub")

    def fake_snapshot_download(**kwargs):
        raise RuntimeError("cache miss")

    fake_hub.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)

    from infra.huggingface_models import resolve_cached_hf_snapshot

    assert resolve_cached_hf_snapshot("pyannote/speaker-diarization-3.1") is None


def test_diarization_loader_uses_cache_resolved_model_reference(
    monkeypatch,
    tmp_path,
):
    _stub_numpy(monkeypatch)
    from pipeline import TranscriptionPipeline
    import pipeline.orchestrator as orchestrator

    cached_snapshot = str(tmp_path / "diarization-snapshot")
    calls = []

    monkeypatch.setattr(
        orchestrator,
        "hf_model_reference",
        lambda repo_id, *, token, purpose: cached_snapshot,
    )

    class FakeLoadedPipeline:
        pass

    class FakePyannotePipeline:
        @classmethod
        def from_pretrained(cls, model_ref, use_auth_token=None):
            calls.append((model_ref, use_auth_token))
            return FakeLoadedPipeline()

    monkeypatch.setattr(
        sys.modules["pyannote.audio"],
        "Pipeline",
        FakePyannotePipeline,
        raising=False,
    )

    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    pipeline.hf_token = "test-token"
    pipeline._diarization = None

    assert pipeline.diarization.__class__ is FakeLoadedPipeline
    assert calls == [(cached_snapshot, "test-token")]


def test_diarization_loader_scopes_torch26_safe_globals(monkeypatch, tmp_path):
    _stub_numpy(monkeypatch)
    from pipeline import TranscriptionPipeline
    import pipeline.orchestrator as orchestrator

    cached_snapshot = str(tmp_path / "diarization-snapshot")
    events = []

    class TorchVersion:
        pass

    torch_version = ModuleType("torch.torch_version")
    torch_version.TorchVersion = TorchVersion
    torch_serialization = ModuleType("torch.serialization")
    pyannote_audio_core = ModuleType("pyannote.audio.core")
    pyannote_audio_core_task = ModuleType("pyannote.audio.core.task")

    class Problem:
        pass

    class Specifications:
        pass

    pyannote_audio_core_task.Problem = Problem
    pyannote_audio_core_task.Specifications = Specifications
    monkeypatch.setitem(sys.modules, "pyannote.audio.core", pyannote_audio_core)
    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio.core.task",
        pyannote_audio_core_task,
    )
    monkeypatch.setattr(
        orchestrator.torch,
        "torch_version",
        torch_version,
        raising=False,
    )
    monkeypatch.setattr(
        orchestrator.torch,
        "serialization",
        torch_serialization,
        raising=False,
    )

    monkeypatch.setattr(
        orchestrator,
        "hf_model_reference",
        lambda repo_id, *, token, purpose: cached_snapshot,
    )

    class FakeSafeGlobals:
        def __init__(self, globals_):
            events.append(("globals", tuple(globals_)))

        def __enter__(self):
            events.append(("enter",))

        def __exit__(self, exc_type, exc, traceback):
            events.append(("exit", exc_type))

    monkeypatch.setattr(
        orchestrator.torch.serialization,
        "safe_globals",
        FakeSafeGlobals,
        raising=False,
    )

    class FakeLoadedPipeline:
        pass

    class FakePyannotePipeline:
        @classmethod
        def from_pretrained(cls, model_ref, use_auth_token=None):
            events.append(("load", model_ref, use_auth_token))
            return FakeLoadedPipeline()

    monkeypatch.setattr(
        sys.modules["pyannote.audio"],
        "Pipeline",
        FakePyannotePipeline,
        raising=False,
    )

    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    pipeline.hf_token = "test-token"
    pipeline._diarization = None

    assert pipeline.diarization.__class__ is FakeLoadedPipeline
    assert events == [
        ("globals", (TorchVersion, Problem, Specifications)),
        ("enter",),
        ("load", cached_snapshot, "test-token"),
        ("exit", None),
    ]


def test_pyannote_safe_globals_falls_back_when_specifications_unavailable(
    monkeypatch,
):
    _stub_numpy(monkeypatch)
    import pipeline.orchestrator as orchestrator

    class TorchVersion:
        pass

    torch_version = ModuleType("torch.torch_version")
    torch_version.TorchVersion = TorchVersion
    monkeypatch.setattr(
        orchestrator.torch,
        "torch_version",
        torch_version,
        raising=False,
    )
    monkeypatch.delitem(sys.modules, "pyannote.audio.core.task", raising=False)
    monkeypatch.delitem(sys.modules, "pyannote.audio.core", raising=False)

    assert orchestrator._trusted_pyannote_checkpoint_globals() == [TorchVersion]


def test_embedding_loader_uses_cache_resolved_model_reference(monkeypatch, tmp_path):
    _stub_numpy(monkeypatch)
    from pipeline import TranscriptionPipeline
    import pipeline.orchestrator as orchestrator

    cached_snapshot = str(tmp_path / "embedding-snapshot")
    calls = []

    monkeypatch.setattr(
        orchestrator,
        "hf_model_reference",
        lambda repo_id, *, token, purpose: cached_snapshot,
    )

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_ref, use_auth_token=None):
            calls.append(("from_pretrained", model_ref, use_auth_token))
            return cls()

        def to(self, device):
            calls.append(("to", device))
            return self

    class FakeInference:
        def __init__(self, model, window):
            calls.append(("inference", model.__class__.__name__, window))

    monkeypatch.setattr(
        sys.modules["pyannote.audio"], "Model", FakeModel, raising=False
    )
    monkeypatch.setattr(
        sys.modules["pyannote.audio"],
        "Inference",
        FakeInference,
        raising=False,
    )

    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    pipeline.hf_token = "test-token"
    pipeline._embedding_model = None

    assert pipeline.embedding_model.__class__ is FakeInference
    assert calls == [
        ("from_pretrained", cached_snapshot, "test-token"),
        ("to", "cpu"),
        ("inference", "FakeModel", "whole"),
    ]
