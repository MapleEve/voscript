"""Provider entrypoints for pipeline-adjacent implementation slots."""

from .asr import PipelineMethodASRProvider, default_asr_provider, transcribe_audio
from .artifacts import InMemoryArtifactsProvider, build_pipeline_artifacts
from .diarization import (
    PipelineMethodDiarizationProvider,
    default_diarization_provider,
    run_diarization,
)
from .embedding import (
    PipelineMethodSpeakerEmbeddingProvider,
    default_speaker_embedding_provider,
    extract_speaker_embeddings,
)
from .enhance import (
    ConditionalDenoiseEnhancer,
    default_audio_enhancer,
    default_enhance_provider,
    maybe_denoise,
)
from .ingest import DefaultIngestProvider, run_ingest
from .normalize import (
    FFmpegInputNormalizer,
    convert_to_wav,
    default_input_normalizer,
    default_normalize_provider,
)
from .postprocess import DefaultPostprocessProvider, run_postprocess
from .punc import DefaultPunctuationProvider, run_punc
from .vad import DefaultVADProvider, run_vad
from .voiceprint_match import (
    DefaultVoiceprintMatchProvider,
    default_voiceprint_match_provider,
    match_speaker_embeddings,
)
from pipeline.registry import (
    available_providers,
    available_stage_slots,
    register_provider,
    resolve_provider,
    unregister_provider,
)

__all__ = [
    "ConditionalDenoiseEnhancer",
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
    "default_speaker_embedding_provider",
    "default_voiceprint_match_provider",
    "extract_speaker_embeddings",
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
