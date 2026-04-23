"""Default provider for speaker embedding extraction."""

from __future__ import annotations

import logging
import os

import numpy as np
import torchaudio

from pipeline.contracts import (
    SpeakerEmbeddingProvider,
    SpeakerEmbeddingRequest,
    SpeakerEmbeddingResult,
)

logger = logging.getLogger(__name__)

# WeSpeaker ResNet34 recommended input window, overridable for ops tuning.
MIN_EMBED_DURATION = float(os.getenv("MIN_EMBED_DURATION", "1.5"))
MAX_EMBED_DURATION = float(os.getenv("MAX_EMBED_DURATION", "10.0"))


def extract_embeddings_for_turns(
    pipeline,
    audio_path: str,
    turns: list[dict[str, object]],
) -> dict[str, np.ndarray]:
    """Extract averaged embeddings for each speaker cluster."""

    info = torchaudio.info(audio_path)
    native_sr = info.sample_rate
    target_sr = 16000
    min_samples = int(MIN_EMBED_DURATION * native_sr)
    max_samples = int(MAX_EMBED_DURATION * native_sr)

    speaker_segments: dict[str, list] = {}
    for turn in turns:
        speaker = turn["speaker"]
        start_sample = int(turn["start"] * native_sr)
        end_sample = int(turn["end"] * native_sr)
        num_frames = end_sample - start_sample

        if num_frames < min_samples:
            continue
        if num_frames > max_samples:
            num_frames = max_samples

        try:
            chunk, chunk_sr = torchaudio.load(
                audio_path,
                frame_offset=start_sample,
                num_frames=num_frames,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load segment %s [%d:%d]: %s",
                speaker,
                start_sample,
                end_sample,
                exc,
            )
            continue

        if chunk_sr != target_sr:
            chunk = torchaudio.functional.resample(chunk, chunk_sr, target_sr)
        if chunk.shape[0] > 1:
            chunk = chunk.mean(dim=0, keepdim=True)

        speaker_segments.setdefault(speaker, []).append(chunk)

    embeddings: dict[str, np.ndarray] = {}
    for speaker, chunks in speaker_segments.items():
        emb_list = []
        chunks.sort(key=lambda chunk: chunk.shape[1], reverse=True)
        for chunk in chunks[:10]:
            emb = pipeline.embedding_model(
                {"waveform": chunk.to(pipeline.device), "sample_rate": target_sr}
            )
            emb_list.append(np.asarray(emb))
        if emb_list:
            embeddings[speaker] = np.mean(emb_list, axis=0)
    return embeddings


class PipelineMethodSpeakerEmbeddingProvider(SpeakerEmbeddingProvider):
    """Extract speaker embeddings through pipeline-owned model resources."""

    def extract_embeddings(
        self, request: SpeakerEmbeddingRequest
    ) -> SpeakerEmbeddingResult:
        return SpeakerEmbeddingResult(
            speaker_embeddings=extract_embeddings_for_turns(
                request.pipeline,
                request.audio_path,
                request.diarization_turns,
            )
        )


default_speaker_embedding_provider = PipelineMethodSpeakerEmbeddingProvider()


__all__ = [
    "PipelineMethodSpeakerEmbeddingProvider",
    "default_speaker_embedding_provider",
    "extract_embeddings_for_turns",
]
