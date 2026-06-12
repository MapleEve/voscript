"""Provider entrypoints for pipeline-adjacent implementation slots."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "CapabilityMatch": ("providers.capabilities", "CapabilityMatch"),
    "ProviderCapability": ("providers.capabilities", "ProviderCapability"),
    "ProviderCapabilityError": ("providers.capabilities", "ProviderCapabilityError"),
    "default_provider_capabilities": (
        "providers.capabilities",
        "default_provider_capabilities",
    ),
    "get_provider_capability": ("providers.capabilities", "get_provider_capability"),
    "match_provider_capability": (
        "providers.capabilities",
        "match_provider_capability",
    ),
    "PipelineMethodASRProvider": ("providers.asr", "PipelineMethodASRProvider"),
    "default_asr_provider": ("providers.asr", "default_asr_provider"),
    "transcribe_audio": ("providers.asr", "transcribe_audio"),
    "InMemoryArtifactsProvider": ("providers.artifacts", "InMemoryArtifactsProvider"),
    "build_pipeline_artifacts": ("providers.artifacts", "build_pipeline_artifacts"),
    "PipelineMethodDiarizationProvider": (
        "providers.diarization",
        "PipelineMethodDiarizationProvider",
    ),
    "default_diarization_provider": (
        "providers.diarization",
        "default_diarization_provider",
    ),
    "run_diarization": ("providers.diarization", "run_diarization"),
    "PipelineMethodSpeakerEmbeddingProvider": (
        "providers.embedding",
        "PipelineMethodSpeakerEmbeddingProvider",
    ),
    "default_speaker_embedding_provider": (
        "providers.embedding",
        "default_speaker_embedding_provider",
    ),
    "extract_speaker_embeddings": (
        "providers.embedding",
        "extract_speaker_embeddings",
    ),
    "ConditionalDenoiseEnhancer": (
        "providers.enhance",
        "ConditionalDenoiseEnhancer",
    ),
    "default_audio_enhancer": ("providers.enhance", "default_audio_enhancer"),
    "default_enhance_provider": ("providers.enhance", "default_enhance_provider"),
    "maybe_denoise": ("providers.enhance", "maybe_denoise"),
    "DefaultIngestProvider": ("providers.ingest", "DefaultIngestProvider"),
    "run_ingest": ("providers.ingest", "run_ingest"),
    "FFmpegInputNormalizer": ("providers.normalize", "FFmpegInputNormalizer"),
    "convert_to_wav": ("providers.normalize", "convert_to_wav"),
    "default_input_normalizer": ("providers.normalize", "default_input_normalizer"),
    "default_normalize_provider": (
        "providers.normalize",
        "default_normalize_provider",
    ),
    "DefaultPostprocessProvider": (
        "providers.postprocess",
        "DefaultPostprocessProvider",
    ),
    "run_postprocess": ("providers.postprocess", "run_postprocess"),
    "DefaultPunctuationProvider": ("providers.punc", "DefaultPunctuationProvider"),
    "run_punc": ("providers.punc", "run_punc"),
    "DefaultVADProvider": ("providers.vad", "DefaultVADProvider"),
    "run_vad": ("providers.vad", "run_vad"),
    "DefaultVoiceprintMatchProvider": (
        "providers.voiceprint_match",
        "DefaultVoiceprintMatchProvider",
    ),
    "default_voiceprint_match_provider": (
        "providers.voiceprint_match",
        "default_voiceprint_match_provider",
    ),
    "match_speaker_embeddings": (
        "providers.voiceprint_match",
        "match_speaker_embeddings",
    ),
    "available_providers": ("providers._registry", "available_providers"),
    "available_stage_slots": ("providers._registry", "available_stage_slots"),
    "register_provider": ("providers._registry", "register_provider"),
    "resolve_provider": ("providers._registry", "resolve_provider"),
    "unregister_provider": ("providers._registry", "unregister_provider"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "ConditionalDenoiseEnhancer",
    "CapabilityMatch",
    "DefaultIngestProvider",
    "DefaultPostprocessProvider",
    "DefaultPunctuationProvider",
    "DefaultVADProvider",
    "DefaultVoiceprintMatchProvider",
    "FFmpegInputNormalizer",
    "InMemoryArtifactsProvider",
    "PipelineMethodASRProvider",
    "PipelineMethodDiarizationProvider",
    "PipelineMethodSpeakerEmbeddingProvider",
    "ProviderCapability",
    "ProviderCapabilityError",
    "available_providers",
    "available_stage_slots",
    "build_pipeline_artifacts",
    "convert_to_wav",
    "default_asr_provider",
    "default_audio_enhancer",
    "default_diarization_provider",
    "default_enhance_provider",
    "default_input_normalizer",
    "default_normalize_provider",
    "default_provider_capabilities",
    "default_speaker_embedding_provider",
    "default_voiceprint_match_provider",
    "extract_speaker_embeddings",
    "get_provider_capability",
    "match_provider_capability",
    "match_speaker_embeddings",
    "maybe_denoise",
    "register_provider",
    "resolve_provider",
    "run_diarization",
    "run_ingest",
    "run_postprocess",
    "run_punc",
    "run_vad",
    "transcribe_audio",
    "unregister_provider",
]
