"""Stable contracts for pluggable pipeline stages and adjacent infra slots."""

from .asr import ASRProvider, ASRRequest, ASRResult
from .artifacts import (
    ARTIFACT_MANIFEST_CATEGORIES,
    ARTIFACT_MANIFEST_VERSION,
    AsyncUploadReader,
    AudioArtifactIndex,
    ArtifactManifestEntry,
    PersistedTranscriptionArtifacts,
    SavedUploadArtifact,
    TranscriptionArtifactStore,
    TranscriptionArtifactWriteRequest,
    UploadPersistenceRequest,
    build_artifact_manifest,
    empty_artifact_manifest,
    normalize_artifact_manifest,
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
    AudioNormalizationError,
    AudioNormalizationRequest,
    AudioNormalizationResult,
    AudioNormalizationTimeoutError,
    InputNormalizationProvider,
)
from .requests import PipelineRequest
from .results import PipelineResult
from .schema import (
    OPTIONAL_FIRST_SCHEMA_POLICY,
    SCHEMA_VERSION_KEY,
    attach_optional_schema_version,
    read_optional_schema_version,
)
from .voiceprint_match import (
    VoiceprintMatchProvider,
    VoiceprintMatchRequest,
    VoiceprintMatchResult,
)

__all__ = [
    "ASRProvider",
    "ASRRequest",
    "ASRResult",
    "ARTIFACT_MANIFEST_CATEGORIES",
    "ARTIFACT_MANIFEST_VERSION",
    "AsyncUploadReader",
    "AudioArtifactIndex",
    "AudioEnhancementProvider",
    "AudioEnhancementRequest",
    "AudioEnhancementResult",
    "AudioNormalizationError",
    "AudioNormalizationRequest",
    "AudioNormalizationResult",
    "AudioNormalizationTimeoutError",
    "ArtifactManifestEntry",
    "DiarizationProvider",
    "DiarizationRequest",
    "DiarizationResult",
    "InputNormalizationProvider",
    "OPTIONAL_FIRST_SCHEMA_POLICY",
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
    "SCHEMA_VERSION_KEY",
    "StageNotFoundError",
    "TranscriptionArtifactStore",
    "TranscriptionArtifactWriteRequest",
    "UploadPersistenceRequest",
    "VoiceprintMatchProvider",
    "VoiceprintMatchRequest",
    "VoiceprintMatchResult",
    "attach_optional_schema_version",
    "build_artifact_manifest",
    "empty_artifact_manifest",
    "normalize_artifact_manifest",
    "read_optional_schema_version",
]
