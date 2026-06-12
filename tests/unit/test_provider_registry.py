"""Unit tests for provider registry and compatibility entrypoints."""

from __future__ import annotations

import sys
import os
from types import ModuleType
from types import SimpleNamespace

import pytest

from pipeline import TranscriptionPipeline
from pipeline.contracts import (
    ASRRequest,
    ASRResult,
    AudioEnhancementResult,
    AudioNormalizationResult,
    DiarizationRequest,
    DiarizationResult,
    PipelineRequest,
    SpeakerEmbeddingRequest,
    SpeakerEmbeddingResult,
    VoiceprintMatchResult,
)
from pipeline.registry import (
    ProviderNotFoundError,
    available_providers,
    is_provider_override,
    register_provider,
    resolve_provider,
    unregister_provider,
)
from providers import maybe_denoise
import providers.asr.default as asr_default
from providers.asr.default import default_asr_provider
import providers.diarization.default as diarization_default
from providers.diarization.default import default_diarization_provider
from providers.embedding import default_speaker_embedding_provider
import providers.embedding.default as embedding_default
import pipeline.orchestrator as orchestrator
from providers.normalize import convert_to_wav


class StubNormalizer:
    def normalize(self, request):
        return AudioNormalizationResult(
            source_path=request.input_path,
            normalized_path=request.input_path.with_suffix(".stub.wav"),
            reused_source=False,
        )


class StubEnhancer:
    def enhance(self, request):
        return AudioEnhancementResult(
            input_path=request.wav_path,
            output_path=request.wav_path.with_suffix(".boost.wav"),
            applied=True,
            model="stub",
        )


def test_default_providers_are_listed_and_resolvable():
    asr_provider = resolve_provider("asr", "default")
    diarization_provider = resolve_provider("diarization", "default")
    embedding_provider = resolve_provider("embedding", "default")
    voiceprint_provider = resolve_provider("voiceprint_match", "default")
    ingest_provider = resolve_provider("ingest", "default")
    normalizer = resolve_provider("normalize", "default")
    enhancer = resolve_provider("enhance", "default")
    vad_provider = resolve_provider("vad", "default")
    punc_provider = resolve_provider("punc", "default")
    postprocess_provider = resolve_provider("postprocess", "default")
    artifacts_provider = resolve_provider("artifacts", "default")

    assert asr_provider.__class__.__name__ == "PipelineMethodASRProvider"
    assert (
        diarization_provider.__class__.__name__ == "PipelineMethodDiarizationProvider"
    )
    assert (
        embedding_provider.__class__.__name__
        == "PipelineMethodSpeakerEmbeddingProvider"
    )
    assert voiceprint_provider.__class__.__name__ == "DefaultVoiceprintMatchProvider"
    assert ingest_provider.__class__.__name__ == "DefaultIngestProvider"
    assert normalizer.__class__.__name__ == "FFmpegInputNormalizer"
    assert enhancer.__class__.__name__ == "ConditionalDenoiseEnhancer"
    assert vad_provider.__class__.__name__ == "DefaultVADProvider"
    assert punc_provider.__class__.__name__ == "DefaultPunctuationProvider"
    assert postprocess_provider.__class__.__name__ == "DefaultPostprocessProvider"
    assert artifacts_provider.__class__.__name__ == "InMemoryArtifactsProvider"
    assert (
        resolve_provider("input_normalization", "default").__class__.__name__
        == "FFmpegInputNormalizer"
    )
    assert (
        resolve_provider("enhancement", "default").__class__.__name__
        == "ConditionalDenoiseEnhancer"
    )
    assert available_providers("ingest") == ("default",)
    assert available_providers("asr") == ("default",)
    assert available_providers("diarization") == ("default",)
    assert available_providers("embedding") == ("default",)
    assert available_providers("voiceprint_match") == ("default",)
    assert available_providers("normalize") == ("default",)
    assert available_providers("enhance") == ("default",)
    assert available_providers("vad") == ("default",)
    assert available_providers("punc") == ("default",)
    assert available_providers("postprocess") == ("default",)
    assert available_providers("artifacts") == ("default",)
    assert available_providers("input_normalization") == ("default",)
    assert available_providers("enhancement") == ("default",)


def test_default_asr_provider_times_materialized_segments(monkeypatch, caplog):
    events = []
    perf_values = iter([10.0, 12.5])

    class FakeSegment:
        start = 0.0
        end = 1.25
        text = " hello "

    class FakeWhisper:
        def transcribe(self, audio_path, **kwargs):
            events.append(("transcribe", audio_path, kwargs["language"]))

            def iter_segments():
                events.append("materialized")
                yield FakeSegment()

            return iter_segments(), SimpleNamespace(language="en")

    pipeline = SimpleNamespace(whisper=FakeWhisper())
    monkeypatch.setattr(
        asr_default.time,
        "perf_counter",
        lambda: events.append("perf") or next(perf_values),
    )

    with caplog.at_level("INFO", logger=asr_default.logger.name):
        result = default_asr_provider.transcribe(
            ASRRequest(
                pipeline=pipeline,
                audio_path="/private/audio.wav",
                language="en",
            )
        )

    assert result.transcription_result["segments"] == [
        {"start": 0.0, "end": 1.25, "text": "hello"}
    ]
    assert events == [
        "perf",
        ("transcribe", "/private/audio.wav", "en"),
        "materialized",
        "perf",
    ]
    assert "asr_processing_timing model=faster-whisper elapsed_s=2.500" in caplog.text
    assert "segment_count=1" in caplog.text
    assert "/private" not in caplog.text


def test_registry_named_overrides_drive_compatibility_helpers(tmp_path):
    input_path = tmp_path / "sample.mp3"
    input_path.write_bytes(b"stub")

    assert is_provider_override("normalize", "stub") is False
    register_provider("normalize", "stub", StubNormalizer())
    register_provider("enhance", "stub", StubEnhancer())
    try:
        assert is_provider_override("input_normalization", "stub") is True
        assert is_provider_override("enhancement", "stub") is True
        normalized = convert_to_wav(input_path, provider_name="stub")
        enhanced = maybe_denoise(normalized, provider_name="stub")
    finally:
        unregister_provider("normalize", "stub")
        unregister_provider("enhance", "stub")

    assert is_provider_override("normalize", "stub") is False
    assert normalized.name == "sample.stub.wav"
    assert enhanced.name == "sample.stub.boost.wav"


def test_unknown_provider_raises_lookup_error():
    with pytest.raises(ProviderNotFoundError):
        resolve_provider("enhance", "missing")


def test_pipeline_request_normalizes_explicit_provider_selection_aliases():
    request = PipelineRequest(
        audio_path="demo.wav",
        provider_selection={
            "input-normalization": "FFmpeg-Basic",
            "enhancement": "DeepFilter-Net",
            "artifacts": "filesystem",
        },
    )

    assert request.provider_for("normalize") == "ffmpeg_basic"
    assert request.provider_for("input_normalization") == "ffmpeg_basic"
    assert request.provider_for("enhance") == "deepfilter_net"
    assert request.provider_for("enhancement") == "deepfilter_net"
    assert request.provider_for("artifacts") == "filesystem"
    assert request.provider_for("voiceprint_match") == "default"


class StubASRProvider:
    def transcribe(self, request):
        return ASRResult(
            transcription_result={
                "segments": [{"start": 0.0, "end": 1.0, "text": "stub"}],
                "language": request.language or "stub",
            }
        )


class StubDiarizationProvider:
    def diarize(self, request):
        return DiarizationResult(
            turns=[{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_STUB"}],
            aligned_segments=[
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "stub",
                    "speaker": "SPEAKER_STUB",
                }
            ],
        )


class StubEmbeddingProvider:
    def extract_embeddings(self, request):
        return SpeakerEmbeddingResult(speaker_embeddings={"SPEAKER_STUB": [0.1, 0.2]})


class StubVoiceprintMatchProvider:
    def match(self, request):
        return VoiceprintMatchResult(
            speaker_map={
                "SPEAKER_STUB": {
                    "matched_id": "spk_stub",
                    "matched_name": "Stub Speaker",
                    "similarity": 0.9876,
                    "embedding_key": "SPEAKER_STUB",
                }
            },
            applied=True,
            threshold=0.7,
            reason="matched",
        )


def test_default_asr_provider_uses_pipeline_whisper_resource():
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    calls = []

    class FakeWhisper:
        def transcribe(self, audio_path, **kwargs):
            calls.append((audio_path, kwargs))
            segments = [SimpleNamespace(start=0.0, end=1.25, text=" hello ")]
            return iter(segments), SimpleNamespace(language="zh")

    pipeline._whisper = FakeWhisper()

    result = default_asr_provider.transcribe(
        ASRRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            language="zh",
            no_repeat_ngram_size=4,
        )
    )

    assert result.transcription_result == {
        "segments": [{"start": 0.0, "end": 1.25, "text": "hello"}],
        "language": "zh",
        "hallucination_guard": {
            "status": "pass",
            "input_segment_count": 1,
            "output_segment_count": 1,
            "removed_segment_count": 0,
            "removed_duration": 0,
        },
    }
    assert calls == [
        (
            "demo.wav",
            {
                "language": "zh",
                "beam_size": 5,
                "vad_filter": True,
                "vad_parameters": {"min_silence_duration_ms": 500},
                "initial_prompt": None,
                "condition_on_previous_text": False,
                "no_repeat_ngram_size": 4,
            },
        )
    ]


def test_default_diarization_provider_uses_pipeline_diarizer_and_alignment(
    monkeypatch, caplog
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []
    perf_values = iter([5.0, 8.0, 10.0, 13.0, 20.0, 21.25])

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            calls.append(("diarizer", audio_path, kwargs))
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        lambda language_code, device: (
            "align-model",
            {"language": language_code, "device": device},
        ),
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "align",
        lambda segments, align_model, align_metadata, audio, device, return_char_alignments=False: {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "text": "hello",
                    "words": [{"start": 0.0, "end": 0.5, "word": "hello"}],
                }
            ]
        },
        raising=False,
    )
    monkeypatch.setattr(
        diarization_default.time,
        "perf_counter",
        lambda: next(perf_values),
    )

    with caplog.at_level("INFO", logger=diarization_default.logger.name):
        result = default_diarization_provider.diarize(
            DiarizationRequest(
                pipeline=pipeline,
                audio_path="demo.wav",
                transcription_result={"segments": [], "language": "en"},
                min_speakers=1,
                max_speakers=2,
            )
        )

    assert calls == [("diarizer", "demo.wav", {"min_speakers": 1, "max_speakers": 2})]
    assert "diarization_processing_timing model=pyannote elapsed_s=3.000" in caplog.text
    assert "Loaded WhisperX alignment model in 3.00s (cold_load=True" in caplog.text
    assert "alignment_processing_timing model=whisperx elapsed_s=1.250" in caplog.text
    assert result.turns == [{"start": 0.0, "end": 1.2, "speaker": "SPEAKER_00"}]
    assert result.aligned_segments == [
        {
            "start": 0.0,
            "end": 1.2,
            "text": "hello",
            "speaker": "SPEAKER_00",
            "words": [{"start": 0.0, "end": 0.5, "word": "hello", "score": 0.0}],
        }
    ]
    assert result.dedup_removed == 0


def test_default_diarization_provider_uses_zh_alignment_override(monkeypatch):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    monkeypatch.setattr(
        diarization_default,
        "WHISPERX_ALIGN_MODEL_MAP",
        {"zh": "safe/zh-align-model"},
    )
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )

    def fake_load_align_model(language_code, device, model_name):
        calls.append(("load_align_model", language_code, device, model_name))
        return "align-model", {"language": language_code, "device": device}

    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        fake_load_align_model,
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "align",
        lambda segments, align_model, align_metadata, audio, device, return_char_alignments=False: {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "text": "你好",
                    "words": [{"start": 0.0, "end": 0.5, "word": "你"}],
                }
            ]
        },
        raising=False,
    )

    result = default_diarization_provider.diarize(
        DiarizationRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            transcription_result={
                "segments": [{"start": 0.0, "end": 1.2, "text": "你好"}],
                "language": "zh",
            },
        )
    )

    assert calls == [("load_align_model", "zh", "cpu", "safe/zh-align-model")]
    assert result.aligned_segments[0]["words"] == [
        {"start": 0.0, "end": 0.5, "word": "你", "score": 0.0}
    ]
    assert result.metadata["alignment"] == {
        "status": "succeeded",
        "language": "zh",
        "model": "safe/zh-align-model",
        "model_source": "override",
        "cache_only": False,
    }


def test_default_diarization_provider_skips_alignment_when_audio_duration_exceeds_budget(
    monkeypatch,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    monkeypatch.setattr(
        diarization_default,
        "audio_duration_seconds",
        lambda audio_path: 7201.0,
    )
    monkeypatch.setattr(
        diarization_default, "WHISPERX_ALIGN_MAX_AUDIO_DURATION_SEC", 7200.0
    )
    monkeypatch.setattr(
        sys.modules["whisperx"],
        "load_audio",
        lambda audio_path: (_ for _ in ()).throw(
            AssertionError("whisperx.load_audio should not run")
        ),
        raising=False,
    )

    result = default_diarization_provider.diarize(
        DiarizationRequest(
            pipeline=pipeline,
            audio_path="long.wav",
            transcription_result={
                "segments": [{"start": 0.0, "end": 1.2, "text": "hello"}],
                "language": "en",
            },
        )
    )

    assert result.metadata["alignment"]["status"] == "skipped"
    assert result.metadata["alignment"]["reason"] == "duration_budget_exceeded"
    assert result.metadata["alignment"]["duration_s"] == 7201.0
    assert result.aligned_segments[0]["speaker"] == "SPEAKER_00"


def test_default_diarization_provider_applies_model_dir_and_cache_only(
    monkeypatch,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    monkeypatch.setattr(
        diarization_default,
        "WHISPERX_ALIGN_MODEL_MAP",
        {"zh": "safe/zh-align-model"},
    )
    monkeypatch.setattr(diarization_default, "WHISPERX_ALIGN_MODEL_DIR", "/cache")
    monkeypatch.setattr(diarization_default, "WHISPERX_ALIGN_CACHE_ONLY", True)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )

    def fake_load_align_model(language_code, device, model_name, model_dir):
        calls.append(
            (
                language_code,
                device,
                model_name,
                model_dir,
                os.environ.get("HF_HUB_OFFLINE"),
                os.environ.get("TRANSFORMERS_OFFLINE"),
            )
        )
        return "align-model", {"language": language_code, "device": device}

    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        fake_load_align_model,
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "align",
        lambda segments, align_model, align_metadata, audio, device, return_char_alignments=False: {
            "segments": segments
        },
        raising=False,
    )

    result = default_diarization_provider.diarize(
        DiarizationRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            transcription_result={
                "segments": [{"start": 0.0, "end": 1.2, "text": "你好"}],
                "language": "zh",
            },
        )
    )

    assert calls == [
        ("zh", "cpu", "safe/zh-align-model", "/cache", "1", "1"),
    ]
    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert os.environ.get("TRANSFORMERS_OFFLINE") is None
    assert result.metadata["alignment"]["cache_only"] is True


def test_default_diarization_provider_attempts_zh_alignment_by_default(monkeypatch):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )

    def fake_load_align_model(language_code, device):
        calls.append(("load_align_model", language_code, device))
        return "align-model", {"language": language_code, "device": device}

    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        fake_load_align_model,
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "align",
        lambda segments, align_model, align_metadata, audio, device, return_char_alignments=False: {
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.2,
                    "text": "你好",
                    "words": [{"start": 0.0, "end": 0.5, "word": "你"}],
                }
            ]
        },
        raising=False,
    )

    result = default_diarization_provider.diarize(
        DiarizationRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            transcription_result={
                "segments": [{"start": 0.0, "end": 1.2, "text": "你好"}],
                "language": "zh",
            },
        )
    )

    assert calls == [("load_align_model", "zh", "cpu")]
    assert result.aligned_segments[0]["words"] == [
        {"start": 0.0, "end": 0.5, "word": "你", "score": 0.0}
    ]
    assert result.metadata["alignment"] == {
        "status": "succeeded",
        "language": "zh",
        "model": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        "model_source": "whisperx_default",
        "cache_only": False,
    }
    assert result.dedup_removed == 0


def test_default_diarization_provider_caches_alignment_model_on_configured_device(
    monkeypatch,
    caplog,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cuda:1"
    calls = []

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    monkeypatch.setattr(diarization_default, "WHISPERX_ALIGN_DEVICE", "cpu")
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )

    def fake_load_align_model(language_code, device):
        calls.append(("load_align_model", language_code, device))
        return object(), {"language": language_code, "device": device}

    def fake_align(
        segments,
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    ):
        calls.append(("align", align_metadata["device"], device))
        return {"segments": segments}

    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        fake_load_align_model,
        raising=False,
    )
    monkeypatch.setattr(whisperx, "align", fake_align, raising=False)

    request = DiarizationRequest(
        pipeline=pipeline,
        audio_path="demo.wav",
        transcription_result={
            "segments": [{"start": 0.0, "end": 1.2, "text": "你好"}],
            "language": "zh",
        },
    )

    with caplog.at_level("INFO", logger=diarization_default.logger.name):
        default_diarization_provider.diarize(request)
        default_diarization_provider.diarize(request)

    assert calls == [
        ("load_align_model", "zh", "cpu"),
        ("align", "cpu", "cpu"),
        ("align", "cpu", "cpu"),
    ]
    assert "Loaded WhisperX alignment model" in caplog.text
    assert "cold_load=True" in caplog.text
    assert "Reusing WhisperX alignment model (hot reuse" in caplog.text
    assert pipeline._alignment_device == "cpu"


def test_default_diarization_provider_skips_zh_alignment_when_explicitly_disabled(
    monkeypatch,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    monkeypatch.setattr(
        diarization_default,
        "WHISPERX_ALIGN_DISABLED_LANGUAGES",
        frozenset({"zh"}),
    )
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: pytest.fail(
            "explicitly disabled zh alignment should not load audio"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        lambda **kwargs: pytest.fail(
            "explicitly disabled zh alignment should not load a model"
        ),
        raising=False,
    )

    result = default_diarization_provider.diarize(
        DiarizationRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            transcription_result={
                "segments": [{"start": 0.0, "end": 1.2, "text": "你好"}],
                "language": "zh",
            },
        )
    )

    assert result.aligned_segments == [
        {
            "start": 0.0,
            "end": 1.2,
            "text": "你好",
            "speaker": "SPEAKER_00",
        }
    ]
    assert result.metadata["alignment"] == {
        "status": "skipped",
        "language": "zh",
        "model": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
        "reason": "language_disabled",
        "actionable_hint": (
            "Remove zh from WHISPERX_ALIGN_DISABLED_LANGUAGES to retry alignment, "
            "or set WHISPERX_ALIGN_MODEL_MAP=zh=<model> for a replacement model."
        ),
    }
    assert result.dedup_removed == 0


def test_default_diarization_provider_classifies_torch_safety_block(
    monkeypatch,
    caplog,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        lambda language_code, device: (_ for _ in ()).throw(
            RuntimeError(
                "Due to a serious vulnerability issue in torch.load, even with "
                "weights_only=True, we now require users to upgrade torch to at "
                "least v2.6 in order to use the function. This version restriction "
                "does not apply when loading files with safetensors."
            )
        ),
        raising=False,
    )

    with caplog.at_level("WARNING", logger="providers.diarization.default"):
        result = default_diarization_provider.diarize(
            DiarizationRequest(
                pipeline=pipeline,
                audio_path="demo.wav",
                transcription_result={
                    "segments": [{"start": 0.0, "end": 1.2, "text": "你好"}],
                    "language": "zh",
                },
            )
        )

    alignment = result.metadata["alignment"]
    assert alignment["status"] == "failed"
    assert alignment["language"] == "zh"
    assert alignment["model"] == "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
    assert alignment["reason"] == "torch_version_blocked"
    assert alignment["error_type"] == "RuntimeError"
    assert "torch>=2.6" in alignment["actionable_hint"]
    assert "safetensors" in alignment["actionable_hint"]
    assert alignment["reason"] != "not_found"
    assert "could not be found" not in caplog.text


def test_default_diarization_provider_sanitizes_alignment_failure_metadata(
    monkeypatch,
    caplog,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"

    class FakeDiarizationResult:
        def itertracks(self, yield_label=False):
            assert yield_label is True
            yield SimpleNamespace(start=0.0, end=1.2), None, "SPEAKER_00"

    class FakeDiarizer:
        def __call__(self, audio_path, **kwargs):
            return FakeDiarizationResult()

    pipeline._diarization = FakeDiarizer()
    whisperx = sys.modules["whisperx"]
    monkeypatch.setattr(
        whisperx,
        "load_audio",
        lambda audio_path: f"audio:{audio_path}",
        raising=False,
    )
    monkeypatch.setattr(
        whisperx,
        "load_align_model",
        lambda language_code, device: (_ for _ in ()).throw(
            RuntimeError("token=secret /private/cache/model.bin")
        ),
        raising=False,
    )

    with caplog.at_level("WARNING", logger="providers.diarization.default"):
        result = default_diarization_provider.diarize(
            DiarizationRequest(
                pipeline=pipeline,
                audio_path="demo.wav",
                transcription_result={
                    "segments": [{"start": 0.0, "end": 1.2, "text": "hello"}],
                    "language": "en",
                },
            )
        )

    assert result.metadata["alignment"] == {
        "status": "failed",
        "language": "en",
        "model": None,
        "reason": "load_or_align_failed",
        "error_type": "RuntimeError",
        "model_source": "whisperx_default",
        "cache_only": False,
        "actionable_hint": (
            "Check WHISPERX_ALIGN_MODEL_MAP, WHISPERX_ALIGN_MODEL_DIR, "
            "WHISPERX_ALIGN_CACHE_ONLY, network access, and model compatibility."
        ),
    }
    assert "token=secret" not in caplog.text
    assert "/private/cache" not in caplog.text


def test_default_embedding_provider_uses_pipeline_embedding_resource(monkeypatch):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []

    class FakeTensor:
        def __init__(self, channels, frames):
            self.shape = (channels, frames)

        def mean(self, dim=0, keepdim=True):
            assert dim == 0
            return FakeTensor(1, self.shape[1])

        def to(self, device):
            calls.append(("to", device, self.shape[1]))
            return self

    class FakeEmbeddingModel:
        def __call__(self, payload):
            calls.append(
                (
                    "embedding_model",
                    payload["sample_rate"],
                    payload["waveform"].shape[1],
                )
            )
            return [float(payload["waveform"].shape[1]), 1.0]

    class FakeInfo:
        sample_rate = 16000

    pipeline._embedding_model = FakeEmbeddingModel()
    monkeypatch.setattr(
        embedding_default.torchaudio, "info", lambda audio_path: FakeInfo()
    )
    monkeypatch.setattr(
        embedding_default.torchaudio,
        "load",
        lambda audio_path, frame_offset, num_frames: (FakeTensor(1, num_frames), 16000),
    )

    result = default_speaker_embedding_provider.extract_embeddings(
        SpeakerEmbeddingRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            diarization_turns=[
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
                {"speaker": "SPEAKER_00", "start": 2.0, "end": 4.0},
                {"speaker": "SPEAKER_SKIP", "start": 0.0, "end": 1.0},
            ],
        )
    )

    assert list(result.speaker_embeddings) == ["SPEAKER_00"]
    assert result.speaker_embeddings["SPEAKER_00"].tolist() == [32000.0, 1.0]
    assert calls == [
        ("to", "cpu", 32000),
        ("embedding_model", 16000, 32000),
        ("to", "cpu", 32000),
        ("embedding_model", 16000, 32000),
    ]


def test_default_embedding_provider_moves_chunks_to_embedding_device(
    monkeypatch, caplog
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cuda:0"
    pipeline._embedding_device = "cuda:1"
    calls = []
    perf_values = iter([29.0, 30.0, 30.75])

    class FakeTensor:
        def __init__(self, channels, frames):
            self.shape = (channels, frames)

        def mean(self, dim=0, keepdim=True):
            assert dim == 0
            return FakeTensor(1, self.shape[1])

        def to(self, device):
            calls.append(("to", device, self.shape[1]))
            return self

    class FakeEmbeddingModel:
        def __call__(self, payload):
            calls.append(("embedding_model", payload["waveform"].shape[1]))
            return [1.0, 2.0]

    class FakeInfo:
        sample_rate = 16000

    pipeline._embedding_model = FakeEmbeddingModel()
    monkeypatch.setattr(
        embedding_default.torchaudio, "info", lambda audio_path: FakeInfo()
    )
    monkeypatch.setattr(
        embedding_default.torchaudio,
        "load",
        lambda audio_path, frame_offset, num_frames: (FakeTensor(1, num_frames), 16000),
    )
    monkeypatch.setattr(
        embedding_default.time,
        "perf_counter",
        lambda: next(perf_values),
    )

    with caplog.at_level("INFO", logger=embedding_default.logger.name):
        result = default_speaker_embedding_provider.extract_embeddings(
            SpeakerEmbeddingRequest(
                pipeline=pipeline,
                audio_path="demo.wav",
                diarization_turns=[
                    {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
                ],
            )
        )

    assert result.speaker_embeddings["SPEAKER_00"].tolist() == [1.0, 2.0]
    assert "embedding_processing_timing model=wespeaker elapsed_s=0.750" in caplog.text
    assert "speaker_count=1" in caplog.text
    assert calls == [
        ("to", "cuda:1", 32000),
        ("embedding_model", 32000),
    ]


def test_default_embedding_provider_prefers_single_soundfile_load(monkeypatch):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []

    class FakeTensor:
        def __init__(self, channels, frames):
            self.shape = (channels, frames)

        def __getitem__(self, key):
            channel_key, frame_key = key
            assert isinstance(channel_key, slice)
            assert channel_key == slice(None)
            start = frame_key.start or 0
            stop = frame_key.stop or self.shape[1]
            return FakeTensor(self.shape[0], max(stop - start, 0))

        def contiguous(self):
            calls.append(("contiguous", self.shape[1]))
            return self

        def mean(self, dim=0, keepdim=True):
            assert dim == 0
            assert keepdim is True
            return FakeTensor(1, self.shape[1])

        def to(self, device):
            calls.append(("to", device, self.shape[1]))
            return self

    class FakeEmbeddingModel:
        def __call__(self, payload):
            calls.append(("embedding_model", payload["waveform"].shape[1]))
            return [float(payload["waveform"].shape[1]), 2.0]

    class FakeArray:
        def __init__(self, shape):
            self.shape = shape

        @property
        def T(self):
            return FakeArray(tuple(reversed(self.shape)))

        def copy(self):
            return self

    pipeline._embedding_model = FakeEmbeddingModel()
    monkeypatch.setattr(
        embedding_default.sf,
        "read",
        lambda audio_path, dtype, always_2d: (
            FakeArray((48000, 1)),
            16000,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        embedding_default.torch,
        "from_numpy",
        lambda data: calls.append(("from_numpy", data.shape)) or FakeTensor(1, 48000),
        raising=False,
    )
    monkeypatch.setattr(
        embedding_default.torchaudio,
        "info",
        lambda audio_path: (_ for _ in ()).throw(
            AssertionError("torchaudio.info should not be used for canonical audio")
        ),
    )
    monkeypatch.setattr(
        embedding_default.torchaudio,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("torchaudio.load should not be used for canonical audio")
        ),
    )

    result = default_speaker_embedding_provider.extract_embeddings(
        SpeakerEmbeddingRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            diarization_turns=[
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
                {"speaker": "SPEAKER_00", "start": 2.0, "end": 4.0},
            ],
        )
    )

    assert result.speaker_embeddings["SPEAKER_00"].tolist() == [32000.0, 2.0]
    assert calls == [
        ("from_numpy", (1, 48000)),
        ("contiguous", 32000),
        ("contiguous", 32000),
        ("to", "cpu", 32000),
        ("embedding_model", 32000),
        ("to", "cpu", 32000),
        ("embedding_model", 32000),
    ]


def test_default_embedding_provider_skips_full_preload_when_duration_exceeds_budget(
    monkeypatch,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []

    class FakeTensor:
        def __init__(self, channels, frames):
            self.shape = (channels, frames)

        def mean(self, dim=0, keepdim=True):
            assert dim == 0
            return FakeTensor(1, self.shape[1])

        def to(self, device):
            calls.append(("to", device, self.shape[1]))
            return self

    class FakeEmbeddingModel:
        def __call__(self, payload):
            calls.append(("embedding_model", payload["waveform"].shape[1]))
            return [float(payload["waveform"].shape[1]), 3.0]

    class FakeInfo:
        sample_rate = 16000

    pipeline._embedding_model = FakeEmbeddingModel()
    monkeypatch.setattr(
        embedding_default, "audio_duration_seconds", lambda path: 1801.0
    )
    monkeypatch.setattr(
        embedding_default,
        "EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC",
        1800.0,
    )
    monkeypatch.setattr(
        embedding_default.sf,
        "read",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("soundfile.read should not preload over-budget audio")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        embedding_default.torchaudio, "info", lambda audio_path: FakeInfo()
    )
    monkeypatch.setattr(
        embedding_default.torchaudio,
        "load",
        lambda audio_path, frame_offset, num_frames: (FakeTensor(1, num_frames), 16000),
    )

    result = default_speaker_embedding_provider.extract_embeddings(
        SpeakerEmbeddingRequest(
            pipeline=pipeline,
            audio_path="long.wav",
            diarization_turns=[
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
            ],
        )
    )

    assert result.speaker_embeddings["SPEAKER_00"].tolist() == [32000.0, 3.0]
    assert calls == [
        ("to", "cpu", 32000),
        ("embedding_model", 32000),
    ]


def test_default_embedding_provider_uses_selected_device_after_first_lazy_load(
    monkeypatch,
):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cuda"
    pipeline._configured_device = "cuda"
    pipeline._embedding_device = None
    pipeline._embedding_model = None
    pipeline.hf_token = None
    calls = []

    class FakeTensor:
        def __init__(self, channels, frames):
            self.shape = (channels, frames)

        def mean(self, dim=0, keepdim=True):
            assert dim == 0
            return FakeTensor(1, self.shape[1])

        def to(self, device):
            calls.append(("chunk_to", device, self.shape[1]))
            return self

    class FakeInfo:
        sample_rate = 16000

    class FakeEmbeddingModel:
        @classmethod
        def from_pretrained(cls, model_ref, use_auth_token=None):
            calls.append(("embedding_load", model_ref, use_auth_token))
            return cls()

        def to(self, device):
            calls.append(("model_to", device))
            return self

    class FakeInference:
        def __init__(self, model, window):
            calls.append(("inference", window))

        def __call__(self, payload):
            calls.append(("embedding_model", payload["waveform"].shape[1]))
            return [1.0, 2.0]

    pyannote_audio = ModuleType("pyannote.audio")
    pyannote_audio.Model = FakeEmbeddingModel
    pyannote_audio.Inference = FakeInference

    monkeypatch.setitem(sys.modules, "pyannote.audio", pyannote_audio)
    monkeypatch.setattr(
        orchestrator,
        "resolve_hf_model_ref",
        lambda repo_id, *, token, purpose: repo_id,
    )
    monkeypatch.setattr(
        orchestrator,
        "select_best_cuda_device",
        lambda configured: calls.append(("select", configured)) or "cuda:1",
    )
    monkeypatch.setattr(
        embedding_default.torchaudio, "info", lambda audio_path: FakeInfo()
    )
    monkeypatch.setattr(
        embedding_default.torchaudio,
        "load",
        lambda audio_path, frame_offset, num_frames: (FakeTensor(1, num_frames), 16000),
    )

    result = default_speaker_embedding_provider.extract_embeddings(
        SpeakerEmbeddingRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            diarization_turns=[
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0},
            ],
        )
    )

    assert result.speaker_embeddings["SPEAKER_00"].tolist() == [1.0, 2.0]
    assert pipeline._embedding_device == "cuda:1"
    assert calls == [
        ("select", "cuda"),
        (
            "embedding_load",
            "pyannote/wespeaker-voxceleb-resnet34-LM",
            None,
        ),
        ("model_to", "cuda:1"),
        ("inference", "whole"),
        ("chunk_to", "cuda:1", 32000),
        ("embedding_model", 32000),
    ]
