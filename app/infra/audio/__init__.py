"""Audio-specific infrastructure helpers."""

from .hash_index import (
    JsonAudioArtifactIndex,
    compute_file_hash,
    default_audio_artifact_index,
    lookup_hash,
    register_hash,
    save_upload_and_hash,
)
from .errors import (
    AudioPathError,
    AudioPathTraversalError,
    InvalidSpeakerLabelError,
    InvalidTranscriptionIdError,
)
from .paths import safe_log_filename, safe_speaker_label, safe_tr_dir
from .tempfiles import cleanup_generated_files
from .metadata import audio_duration_seconds

__all__ = [
    "AudioPathError",
    "AudioPathTraversalError",
    "audio_duration_seconds",
    "cleanup_generated_files",
    "JsonAudioArtifactIndex",
    "compute_file_hash",
    "default_audio_artifact_index",
    "InvalidSpeakerLabelError",
    "InvalidTranscriptionIdError",
    "lookup_hash",
    "register_hash",
    "safe_log_filename",
    "safe_speaker_label",
    "safe_tr_dir",
    "save_upload_and_hash",
]
