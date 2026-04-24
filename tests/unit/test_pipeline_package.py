"""Unit tests for the pipeline package entrypoints and orchestration."""

from pipeline import TranscriptionPipeline
from pipeline.contracts import (
    ASRResult,
    AudioEnhancementResult,
    AudioNormalizationResult,
    DiarizationResult,
    SpeakerEmbeddingResult,
)
from pipeline.registry import register_provider, unregister_provider


def test_package_exports_transcription_pipeline():
    assert TranscriptionPipeline.__module__ == "pipeline.orchestrator"


def test_process_keeps_stage_order_and_result_shape(monkeypatch):
    pipeline = TranscriptionPipeline.__new__(TranscriptionPipeline)

    calls = []

    class StubNormalizeProvider:
        def normalize(self, request):
            return AudioNormalizationResult(
                source_path=request.input_path,
                normalized_path=request.input_path,
                reused_source=True,
            )

    class StubEnhanceProvider:
        def enhance(self, request):
            return AudioEnhancementResult(
                input_path=request.wav_path,
                output_path=request.wav_path,
                applied=False,
                model="stub",
            )

    class StubASRProvider:
        def transcribe(self, request):
            calls.append(("asr", request.audio_path, request.language, request.no_repeat_ngram_size))
            return ASRResult(
                transcription_result={
                    "segments": [{"start": 0.0, "end": 1.0, "text": " hi "}],
                    "language": "zh",
                }
            )

    class StubDiarizationProvider:
        def diarize(self, request):
            calls.append(("diarization", request.audio_path, request.min_speakers, request.max_speakers))
            return DiarizationResult(
                turns=[{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
                aligned_segments=[
                    {"start": 0.0, "end": 0.8, "text": "嗯", "speaker": "SPEAKER_00"},
                    {"start": 0.9, "end": 1.4, "text": "嗯", "speaker": "SPEAKER_00"},
                    {"start": 1.5, "end": 4.0, "text": "继续", "speaker": "SPEAKER_00"},
                ],
                dedup_removed=0,
            )

    class StubEmbeddingProvider:
        def extract_embeddings(self, request):
            calls.append(("embedding", request.audio_path, request.diarization_turns))
            return SpeakerEmbeddingResult(
                speaker_embeddings={"SPEAKER_00": [0.1, 0.2]}
            )

    register_provider("normalize", "stub", StubNormalizeProvider())
    register_provider("enhance", "stub", StubEnhanceProvider())
    register_provider("asr", "stub", StubASRProvider())
    register_provider("diarization", "stub", StubDiarizationProvider())
    register_provider("embedding", "stub", StubEmbeddingProvider())
    try:
        result = pipeline.process(
            "clean.wav",
            raw_audio_path="raw.wav",
            language="zh",
            min_speakers=1,
            max_speakers=2,
            no_repeat_ngram_size=3,
            provider_selection={
                "normalize": "stub",
                "enhance": "stub",
                "asr": "stub",
                "diarization": "stub",
                "embedding": "stub",
            },
        )
    finally:
        unregister_provider("normalize", "stub")
        unregister_provider("enhance", "stub")
        unregister_provider("asr", "stub")
        unregister_provider("diarization", "stub")
        unregister_provider("embedding", "stub")

    assert calls == [
        ("asr", "clean.wav", "zh", 3),
        ("diarization", "clean.wav", 1, 2),
        (
            "embedding",
            "raw.wav",
            [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
        ),
    ]
    assert result == {
        "segments": [
            {"start": 0.0, "end": 0.8, "text": "嗯", "speaker": "SPEAKER_00"},
            {"start": 0.9, "end": 1.4, "text": "嗯", "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 4.0, "text": "继续", "speaker": "SPEAKER_00"},
        ],
        "speaker_embeddings": {"SPEAKER_00": [0.1, 0.2]},
        "unique_speakers": ["SPEAKER_00"],
    }
