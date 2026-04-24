"""Default provider for ASR via pipeline-owned Whisper resources."""

from __future__ import annotations

import logging

from pipeline.contracts import ASRProvider, ASRRequest, ASRResult

logger = logging.getLogger(__name__)


def run_faster_whisper_asr(
    pipeline,
    audio_path: str,
    *,
    language: str | None = None,
    no_repeat_ngram_size: int | None = None,
) -> dict[str, object]:
    """Run faster-whisper through the pipeline resource facade."""

    lang_arg = language if language else None
    initial_prompt = (
        "以下是普通话的对话，请以简体中文输出。" if lang_arg is None else None
    )
    logger.info(
        "Starting faster-whisper transcription (language=%s)",
        lang_arg or "auto",
    )

    whisper_kwargs = dict(
        language=lang_arg,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        initial_prompt=initial_prompt,
    )
    if no_repeat_ngram_size and no_repeat_ngram_size >= 3:
        whisper_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    segments_iter, info = pipeline.whisper.transcribe(audio_path, **whisper_kwargs)
    segments = [
        {
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "text": segment.text.strip(),
        }
        for segment in segments_iter
    ]
    detected_language = info.language
    logger.info(
        "Transcription done: %d segments, language=%s",
        len(segments),
        detected_language,
    )
    return {"segments": segments, "language": detected_language}


class PipelineMethodASRProvider(ASRProvider):
    """Run ASR through the current pipeline resource facade."""

    def transcribe(self, request: ASRRequest) -> ASRResult:
        return ASRResult(
            transcription_result=run_faster_whisper_asr(
                request.pipeline,
                request.audio_path,
                language=request.language,
                no_repeat_ngram_size=request.no_repeat_ngram_size,
            )
        )


default_asr_provider = PipelineMethodASRProvider()


__all__ = [
    "PipelineMethodASRProvider",
    "default_asr_provider",
    "run_faster_whisper_asr",
]
