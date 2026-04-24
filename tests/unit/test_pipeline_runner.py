"""Unit tests for stable pipeline stage slots and runner orchestration."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from pipeline import TranscriptionPipeline
from pipeline.contracts import (
    ASRResult,
    AudioEnhancementResult,
    AudioNormalizationResult,
    DiarizationResult,
    PipelineContext,
    PipelineResult,
    SpeakerEmbeddingResult,
    VoiceprintMatchResult,
)
from pipeline.registry import (
    available_stage_slots,
    register_provider,
    resolve_stage,
    unregister_provider,
)
from pipeline.runner import DEFAULT_STAGE_ORDER, PipelineRequest, PipelineRunner
from pipeline.stages import (
    available_stage_slots as available_stage_slots_compat,
    resolve_stage as resolve_stage_compat,
)


def test_stage_slots_publish_stable_order_and_callable_entrypoints():
    expected = (
        "ingest",
        "normalize",
        "enhance",
        "vad",
        "asr",
        "diarization",
        "embedding",
        "voiceprint_match",
        "punc",
        "postprocess",
        "artifacts",
    )

    assert available_stage_slots() == expected
    assert available_stage_slots_compat() == expected
    assert DEFAULT_STAGE_ORDER == expected
    assert callable(resolve_stage("ingest"))
    assert callable(resolve_stage("asr"))
    assert callable(resolve_stage("artifacts"))
    assert callable(resolve_stage_compat("ingest"))


def test_runner_builds_shared_pipeline_context_type():
    context = PipelineRunner().build_context(
        SimpleNamespace(),
        PipelineRequest(audio_path="sample.wav"),
    )

    assert isinstance(context, PipelineContext)


def test_runner_executes_stable_stage_order_and_builds_result(monkeypatch):
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
        request = PipelineRequest(
            audio_path="clean.wav",
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

        context = PipelineRunner().run_context(pipeline, request)
    finally:
        unregister_provider("normalize", "stub")
        unregister_provider("enhance", "stub")
        unregister_provider("asr", "stub")
        unregister_provider("diarization", "stub")
        unregister_provider("embedding", "stub")

    assert context.metadata["executed_stages"] == list(DEFAULT_STAGE_ORDER)
    assert context.metadata["selected_providers"]["normalize"] == "stub"
    assert context.metadata["selected_providers"]["embedding"] == "stub"
    assert calls == [
        ("asr", "clean.wav", "zh", 3),
        ("diarization", "clean.wav", 1, 2),
        (
            "embedding",
            "raw.wav",
            [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
        ),
    ]
    assert context.to_result() == {
        "segments": [
            {"start": 0.0, "end": 0.8, "text": "嗯", "speaker": "SPEAKER_00"},
            {"start": 0.9, "end": 1.4, "text": "嗯", "speaker": "SPEAKER_00"},
            {"start": 1.5, "end": 4.0, "text": "继续", "speaker": "SPEAKER_00"},
        ],
        "speaker_embeddings": {"SPEAKER_00": [0.1, 0.2]},
        "unique_speakers": ["SPEAKER_00"],
    }


def test_runner_dispatches_pipeline_steps_through_provider_registry():
    calls = []

    class StubNormalizeProvider:
        def normalize(self, request):
            calls.append(("normalize", str(request.input_path)))
            return AudioNormalizationResult(
                source_path=request.input_path,
                normalized_path=request.input_path,
                reused_source=True,
            )

    class StubEnhanceProvider:
        def enhance(self, request):
            calls.append(("enhance", str(request.wav_path), request.model))
            return AudioEnhancementResult(
                input_path=request.wav_path,
                output_path=request.wav_path,
                applied=False,
                model="stub",
            )

    class StubASRProvider:
        def transcribe(self, request):
            calls.append(("asr", request.audio_path, request.language))
            return ASRResult(
                transcription_result={
                    "segments": [{"start": 0.0, "end": 1.0, "text": "stub"}],
                    "language": "stub",
                }
            )

    class StubDiarizationProvider:
        def diarize(self, request):
            calls.append(("diarization", request.audio_path, request.min_speakers))
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
                dedup_removed=0,
            )

    class StubEmbeddingProvider:
        def extract_embeddings(self, request):
            calls.append(("embedding", request.audio_path, len(request.diarization_turns)))
            return SpeakerEmbeddingResult(
                speaker_embeddings={"SPEAKER_STUB": [0.1, 0.2]}
            )

    class StubVoiceprintMatchProvider:
        def match(self, request):
            calls.append(("voiceprint_match", tuple(sorted(request.speaker_embeddings))))
            return VoiceprintMatchResult(
                speaker_map={
                    "SPEAKER_STUB": {
                        "matched_id": "spk_stub",
                        "matched_name": "Stub Speaker",
                        "similarity": 0.95,
                        "embedding_key": "SPEAKER_STUB",
                    }
                },
                applied=True,
                threshold=0.7,
                reason="matched",
            )

    register_provider("normalize", "stub", StubNormalizeProvider())
    register_provider("enhance", "stub", StubEnhanceProvider())
    register_provider("asr", "stub", StubASRProvider())
    register_provider("diarization", "stub", StubDiarizationProvider())
    register_provider("embedding", "stub", StubEmbeddingProvider())
    register_provider("voiceprint_match", "stub", StubVoiceprintMatchProvider())
    try:
        request = PipelineRequest(
            audio_path="clean.wav",
            raw_audio_path="raw.wav",
            language="zh",
            min_speakers=1,
            voiceprint_db=object(),
            voiceprint_threshold=0.7,
            provider_selection={
                "normalize": "stub",
                "enhance": "stub",
                "asr": "stub",
                "diarization": "stub",
                "embedding": "stub",
                "voiceprint_match": "stub",
            },
        )
        context = PipelineRunner().run_context(SimpleNamespace(), request)
    finally:
        unregister_provider("normalize", "stub")
        unregister_provider("enhance", "stub")
        unregister_provider("asr", "stub")
        unregister_provider("diarization", "stub")
        unregister_provider("embedding", "stub")
        unregister_provider("voiceprint_match", "stub")

    assert calls == [
        ("normalize", "clean.wav"),
        ("enhance", "clean.wav", None),
        ("asr", "clean.wav", "zh"),
        ("diarization", "clean.wav", 1),
        ("embedding", "raw.wav", 1),
        ("voiceprint_match", ("SPEAKER_STUB",)),
    ]
    assert context.voiceprint_matches == {
        "SPEAKER_STUB": {
            "matched_id": "spk_stub",
            "matched_name": "Stub Speaker",
            "similarity": 0.95,
            "embedding_key": "SPEAKER_STUB",
        }
    }
    assert context.metadata["voiceprint_match"] == {
        "applied": True,
        "speaker_count": 1,
        "reason": "matched",
        "threshold": 0.7,
    }


def test_runner_persists_artifacts_and_cleans_generated_audio(tmp_path):
    audio_path = tmp_path / "sample.mp3"
    audio_path.write_bytes(b"stub-audio")
    calls = []

    class StubNormalizeProvider:
        def normalize(self, request):
            normalized = request.input_path.with_suffix(".wav")
            normalized.write_bytes(b"normalized")
            calls.append(("normalize", str(normalized)))
            return AudioNormalizationResult(
                source_path=request.input_path,
                normalized_path=normalized,
                reused_source=False,
            )

    class StubEnhanceProvider:
        def enhance(self, request):
            enhanced = request.wav_path.with_suffix(".denoised.wav")
            enhanced.write_bytes(b"enhanced")
            calls.append(("enhance", str(enhanced)))
            return AudioEnhancementResult(
                input_path=request.wav_path,
                output_path=enhanced,
                applied=True,
                model="stub",
            )

    class StubASRProvider:
        def transcribe(self, request):
            calls.append(("asr", request.audio_path))
            return ASRResult(
                transcription_result={
                    "segments": [{"start": 0.0, "end": 1.0, "text": "stub"}],
                    "language": "zh",
                }
            )

    class StubDiarizationProvider:
        def diarize(self, request):
            calls.append(("diarization", request.audio_path))
            return DiarizationResult(
                turns=[{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
                aligned_segments=[
                    {
                        "start": 0.0,
                        "end": 1.0,
                        "text": "stub",
                        "speaker": "SPEAKER_00",
                    }
                ],
                dedup_removed=0,
            )

    class StubEmbeddingProvider:
        def extract_embeddings(self, request):
            calls.append(("embedding", request.audio_path))
            return SpeakerEmbeddingResult(
                speaker_embeddings={"SPEAKER_00": [0.1, 0.2]}
            )

    class StubVoiceprintMatchProvider:
        def match(self, request):
            calls.append(("voiceprint_match", tuple(sorted(request.speaker_embeddings))))
            return VoiceprintMatchResult(
                speaker_map={
                    "SPEAKER_00": {
                        "matched_id": "spk_demo",
                        "matched_name": "Demo Speaker",
                        "similarity": 0.91,
                        "embedding_key": "SPEAKER_00",
                    }
                },
                applied=True,
                threshold=0.7,
                reason="matched",
            )

    register_provider("normalize", "stub", StubNormalizeProvider())
    register_provider("enhance", "stub", StubEnhanceProvider())
    register_provider("asr", "stub", StubASRProvider())
    register_provider("diarization", "stub", StubDiarizationProvider())
    register_provider("embedding", "stub", StubEmbeddingProvider())
    register_provider("voiceprint_match", "stub", StubVoiceprintMatchProvider())
    try:
        request = PipelineRequest(
            audio_path=str(audio_path),
            language="zh",
            voiceprint_db=object(),
            voiceprint_threshold=0.7,
            artifact_dir=tmp_path / "transcriptions" / "tr_demo",
            provider_selection={
                "normalize": "stub",
                "enhance": "stub",
                "asr": "stub",
                "diarization": "stub",
                "embedding": "stub",
                "voiceprint_match": "stub",
            },
        )
        context = PipelineRunner().run_context(SimpleNamespace(), request)
    finally:
        unregister_provider("normalize", "stub")
        unregister_provider("enhance", "stub")
        unregister_provider("asr", "stub")
        unregister_provider("diarization", "stub")
        unregister_provider("embedding", "stub")
        unregister_provider("voiceprint_match", "stub")

    result = context.to_result()
    result_path = tmp_path / "transcriptions" / "tr_demo" / "result.json"
    emb_path = tmp_path / "transcriptions" / "tr_demo" / "emb_SPEAKER_00.npy"

    assert calls == [
        ("normalize", str(audio_path.with_suffix(".wav"))),
        ("enhance", str(audio_path.with_suffix(".denoised.wav"))),
        ("asr", str(audio_path.with_suffix(".denoised.wav"))),
        ("diarization", str(audio_path.with_suffix(".denoised.wav"))),
        ("embedding", str(audio_path.with_suffix(".wav"))),
        ("voiceprint_match", ("SPEAKER_00",)),
    ]
    assert result["transcription"]["id"] == "tr_demo"
    assert result["transcription"]["speaker_map"]["SPEAKER_00"]["matched_id"] == "spk_demo"
    assert result["artifact_paths"]["result_path"] == str(result_path)
    assert result_path.exists()
    assert emb_path.exists()
    assert not audio_path.with_suffix(".wav").exists()
    assert not audio_path.with_suffix(".denoised.wav").exists()


def test_runner_uses_explicit_artifacts_provider_selection():
    class StubArtifactsProvider:
        def build(self, context):
            return PipelineResult(
                segments=[],
                speaker_embeddings={},
                unique_speakers=[],
                transcription={"id": "tr_selected"},
                artifact_paths={"result_path": "memory://tr_selected/result.json"},
            )

    register_provider("artifacts", "memory_stub", StubArtifactsProvider())
    try:
        request = PipelineRequest(
            audio_path="demo.wav",
            provider_selection={"artifacts": "memory_stub"},
        )
        context = PipelineRunner(stage_order=("artifacts",)).run_context(
            SimpleNamespace(),
            request,
        )
    finally:
        unregister_provider("artifacts", "memory_stub")

    result = context.to_result()

    assert context.metadata["selected_providers"]["artifacts"] == "memory_stub"
    assert result["transcription"]["id"] == "tr_selected"
    assert result["artifact_paths"]["result_path"] == "memory://tr_selected/result.json"


def test_runner_cleans_temporary_paths_and_keeps_metadata_on_stage_failure(
    tmp_path, monkeypatch
):
    source = tmp_path / "input.mp3"
    source.write_bytes(b"audio")
    normalized = tmp_path / "input.wav"
    enhanced = tmp_path / "input.denoised.wav"

    request = PipelineRequest(
        audio_path=str(source),
        provider_selection={
            "normalize": "norm-stub",
            "enhance": "enhance-stub",
        },
    )
    runner = PipelineRunner(
        stage_order=("ingest", "normalize", "enhance"),
        stage_overrides={
            "ingest": lambda context: context.metadata.__setitem__(
                "ingest",
                {"status": "ready"},
            ),
            "normalize": lambda context: _stage_write_temp(
                context,
                normalized,
                metadata_key="normalize",
            ),
            "enhance": lambda context: _stage_fail_after_temp(
                context,
                enhanced,
                metadata_key="enhance",
            ),
        },
    )
    context = runner.build_context(SimpleNamespace(), request)
    monkeypatch.setattr(runner, "build_context", lambda pipeline, request: context)

    with pytest.raises(RuntimeError, match="enhance exploded"):
        runner.run_context(SimpleNamespace(), request)

    assert not normalized.exists()
    assert not enhanced.exists()
    assert context.metadata["executed_stages"] == ["ingest", "normalize", "enhance"]
    assert context.metadata["selected_providers"] == {
        "ingest": "default",
        "normalize": "norm_stub",
        "enhance": "enhance_stub",
    }
    assert context.metadata["normalize"]["temporary_path"] == str(normalized)
    assert context.metadata["enhance"]["temporary_path"] == str(enhanced)


def _stage_write_temp(context, path: Path, *, metadata_key: str) -> None:
    path.write_bytes(metadata_key.encode("utf-8"))
    context.temporary_paths.append(path)
    context.working_audio_path = str(path)
    context.metadata[metadata_key] = {
        "status": "prepared",
        "temporary_path": str(path),
    }


def _stage_fail_after_temp(context, path: Path, *, metadata_key: str) -> None:
    _stage_write_temp(context, path, metadata_key=metadata_key)
    context.metadata[metadata_key]["status"] = "failing"
    raise RuntimeError("enhance exploded")
