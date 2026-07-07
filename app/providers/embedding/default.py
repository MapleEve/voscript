"""Default provider for speaker embedding extraction."""

from __future__ import annotations

import logging
import time

import numpy as np
import soundfile as sf
import torch
import torchaudio

from config import (
    EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC,
    MAX_EMBED_DURATION,
    MIN_EMBED_DURATION,
)
from infra.audio import audio_duration_seconds
from pipeline.contracts import (
    SpeakerEmbeddingProvider,
    SpeakerEmbeddingRequest,
    SpeakerEmbeddingResult,
)

logger = logging.getLogger(__name__)


def _should_preload_full_waveform(audio_path: str) -> bool:
    duration_s = audio_duration_seconds(audio_path)
    if (
        EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC > 0
        and duration_s is not None
        and duration_s > EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC
    ):
        logger.info(
            "embedding_full_audio_preload_skipped reason=duration_budget_exceeded "
            "duration_s=%.3f max_duration_s=%.3f",
            duration_s,
            EMBEDDING_PRELOAD_MAX_AUDIO_DURATION_SEC,
        )
        return False
    return True


def _load_full_waveform(audio_path: str):
    """Load normalized audio once with libsndfile to avoid per-turn torch decode."""

    load_started = time.perf_counter()
    data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    waveform = torch.from_numpy(data.T.copy())
    logger.info(
        "embedding_audio_load_timing backend=soundfile elapsed_s=%.3f sample_rate=%d channels=%d frames=%d",
        time.perf_counter() - load_started,
        sample_rate,
        waveform.shape[0],
        waveform.shape[1],
    )
    return waveform, sample_rate


def extract_embeddings_for_turns(
    pipeline,
    audio_path: str,
    turns: list[dict[str, object]],
) -> dict[str, np.ndarray]:
    """Extract averaged embeddings for each speaker cluster."""

    waveform = None
    if _should_preload_full_waveform(audio_path):
        try:
            waveform, native_sr = _load_full_waveform(audio_path)
        except Exception as exc:
            logger.warning(
                "Falling back to torchaudio segment loading for embedding audio: %s",
                exc,
            )
            info = torchaudio.info(audio_path)
            native_sr = info.sample_rate
    else:
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

        if waveform is not None:
            chunk = waveform[:, start_sample : start_sample + num_frames].contiguous()
            chunk_sr = native_sr
        else:
            try:
                chunk, chunk_sr = torchaudio.load(
                    audio_path,
                    frame_offset=start_sample,
                    num_frames=num_frames,
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load embedding audio segment [%d:%d]: %s",
                    start_sample,
                    end_sample,
                    exc,
                )
                continue
        if chunk.shape[1] <= 0:
            continue

        if chunk_sr != target_sr:
            chunk = torchaudio.functional.resample(chunk, chunk_sr, target_sr)
        if chunk.shape[0] > 1:
            chunk = chunk.mean(dim=0, keepdim=True)

        speaker_segments.setdefault(speaker, []).append(chunk)

    embeddings: dict[str, np.ndarray] = {}
    model_processing_elapsed_s = 0.0
    processed_chunk_count = 0
    for speaker, chunks in speaker_segments.items():
        emb_list = []
        chunks.sort(key=lambda chunk: chunk.shape[1], reverse=True)
        for chunk in chunks[:10]:
            embedding_model = pipeline.embedding_model
            embedding_device = getattr(pipeline, "embedding_device", pipeline.device)
            processing_started = time.perf_counter()
            emb = embedding_model(
                {"waveform": chunk.to(embedding_device), "sample_rate": target_sr}
            )
            model_processing_elapsed_s += time.perf_counter() - processing_started
            processed_chunk_count += 1
            emb_list.append(np.asarray(emb))
        if emb_list:
            embeddings[speaker] = np.mean(emb_list, axis=0)
    logger.info(
        "embedding_processing_timing model=wespeaker elapsed_s=%.3f device=%s speaker_count=%d chunk_count=%d",
        model_processing_elapsed_s,
        getattr(pipeline, "embedding_device", getattr(pipeline, "device", "")),
        len(embeddings),
        processed_chunk_count,
    )
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
