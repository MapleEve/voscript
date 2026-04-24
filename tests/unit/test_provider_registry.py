"""Unit tests for provider registry and compatibility entrypoints."""

from __future__ import annotations

import sys
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
    register_provider,
    resolve_provider,
    unregister_provider,
)
from providers import maybe_denoise
from providers.asr.default import default_asr_provider
from providers.diarization.default import default_diarization_provider
from providers.embedding import default_speaker_embedding_provider
import providers.embedding.default as embedding_default
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
    assert diarization_provider.__class__.__name__ == "PipelineMethodDiarizationProvider"
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
    assert resolve_provider("input_normalization", "default").__class__.__name__ == "FFmpegInputNormalizer"
    assert resolve_provider("enhancement", "default").__class__.__name__ == "ConditionalDenoiseEnhancer"
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


def test_registry_named_overrides_drive_compatibility_helpers(tmp_path):
    input_path = tmp_path / "sample.mp3"
    input_path.write_bytes(b"stub")

    register_provider("normalize", "stub", StubNormalizer())
    register_provider("enhance", "stub", StubEnhancer())
    try:
        normalized = convert_to_wav(input_path, provider_name="stub")
        enhanced = maybe_denoise(normalized, provider_name="stub")
    finally:
        unregister_provider("normalize", "stub")
        unregister_provider("enhance", "stub")

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
        return SpeakerEmbeddingResult(
            speaker_embeddings={"SPEAKER_STUB": [0.1, 0.2]}
        )


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
                "no_repeat_ngram_size": 4,
            },
        )
    ]


def test_default_diarization_provider_uses_pipeline_diarizer_and_alignment(monkeypatch):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)
    pipeline.device = "cpu"
    calls = []

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
        lambda language_code, device: ("align-model", {"language": language_code, "device": device}),
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

    result = default_diarization_provider.diarize(
        DiarizationRequest(
            pipeline=pipeline,
            audio_path="demo.wav",
            transcription_result={"segments": [], "language": "zh"},
            min_speakers=1,
            max_speakers=2,
        )
    )

    assert calls == [("diarizer", "demo.wav", {"min_speakers": 1, "max_speakers": 2})]
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
            calls.append(("embedding_model", payload["sample_rate"], payload["waveform"].shape[1]))
            return [float(payload["waveform"].shape[1]), 1.0]

    class FakeInfo:
        sample_rate = 16000

    pipeline._embedding_model = FakeEmbeddingModel()
    monkeypatch.setattr(embedding_default.torchaudio, "info", lambda audio_path: FakeInfo())
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
