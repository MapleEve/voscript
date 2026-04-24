"""Stable contracts for pluggable pipeline stages and adjacent infra slots."""

from .asr import ASRProvider, ASRRequest, ASRResult
from .artifacts import (
    AsyncUploadReader,
    AudioArtifactIndex,
    PersistedTranscriptionArtifacts,
    SavedUploadArtifact,
    TranscriptionArtifactStore,
    TranscriptionArtifactWriteRequest,
    UploadPersistenceRequest,
)
from .context import PipelineContext
from .diarization import (
    DiarizationProvider,
    DiarizationRequest,
    DiarizationResult,
)
from .enhance import (
    AudioEnhancementProvider,
    AudioEnhancementRequest,
    AudioEnhancementResult,
)
from .embedding import (
    SpeakerEmbeddingProvider,
    SpeakerEmbeddingRequest,
    SpeakerEmbeddingResult,
)
from .errors import (
    PipelineLookupError,
    ProviderNotFoundError,
    StageNotFoundError,
)
from .normalize import (
    AudioNormalizationRequest,
    AudioNormalizationResult,
    InputNormalizationProvider,
)
from .requests import PipelineRequest
from .results import PipelineResult
from .voiceprint_match import (
    VoiceprintMatchProvider,
    VoiceprintMatchRequest,
    VoiceprintMatchResult,
)

__all__ = [
    "ASRProvider",
    "ASRRequest",
    "ASRResult",
    "AsyncUploadReader",
    "AudioArtifactIndex",
    "AudioEnhancementProvider",
    "AudioEnhancementRequest",
    "AudioEnhancementResult",
    "AudioNormalizationRequest",
    "AudioNormalizationResult",
    "DiarizationProvider",
    "DiarizationRequest",
    "DiarizationResult",
    "InputNormalizationProvider",
    "PersistedTranscriptionArtifacts",
    "PipelineContext",
    "PipelineLookupError",
    "PipelineRequest",
    "PipelineResult",
    "SavedUploadArtifact",
    "SpeakerEmbeddingProvider",
    "SpeakerEmbeddingRequest",
    "SpeakerEmbeddingResult",
    "ProviderNotFoundError",
    "StageNotFoundError",
    "TranscriptionArtifactStore",
    "TranscriptionArtifactWriteRequest",
    "UploadPersistenceRequest",
    "VoiceprintMatchProvider",
    "VoiceprintMatchRequest",
    "VoiceprintMatchResult",
]
