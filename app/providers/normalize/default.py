"""Default ffmpeg-backed provider for input audio normalization."""

from __future__ import annotations

import logging
import subprocess

from fastapi import HTTPException

from config import FFMPEG_TIMEOUT_SEC
from pipeline.contracts import (
    AudioNormalizationRequest,
    AudioNormalizationResult,
    InputNormalizationProvider,
)

logger = logging.getLogger(__name__)


class FFmpegInputNormalizer(InputNormalizationProvider):
    """Normalize uploads into the WAV shape expected by the pipeline."""

    def normalize(
        self, request: AudioNormalizationRequest
    ) -> AudioNormalizationResult:
        input_path = request.input_path
        target_suffix = f".{request.target_format.lstrip('.').lower()}"
        normalized_path = input_path.with_suffix(target_suffix)

        if input_path.suffix.lower() == target_suffix:
            return AudioNormalizationResult(
                source_path=input_path,
                normalized_path=input_path,
                reused_source=True,
            )

        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-v",
                    "error",
                    "-i",
                    str(input_path),
                    "-ar",
                    str(request.target_sample_rate),
                    "-ac",
                    str(request.target_channels),
                    "-f",
                    request.target_format,
                    "--",
                    str(normalized_path),
                ],
                check=True,
                timeout=FFMPEG_TIMEOUT_SEC,
            )
        except subprocess.TimeoutExpired:
            normalized_path.unlink(missing_ok=True)
            logger.error(
                "ffmpeg timed out after %ds on %s",
                FFMPEG_TIMEOUT_SEC,
                input_path.name,
            )
            raise HTTPException(504, f"ffmpeg timed out after {FFMPEG_TIMEOUT_SEC}s")

        return AudioNormalizationResult(
            source_path=input_path,
            normalized_path=normalized_path,
            reused_source=False,
        )


default_normalize_provider = FFmpegInputNormalizer()
default_input_normalizer = default_normalize_provider


__all__ = [
    "FFmpegInputNormalizer",
    "default_input_normalizer",
    "default_normalize_provider",
]
