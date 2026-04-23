"""Stable slot for automatic speech recognition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from providers.asr import transcribe_audio

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Run the current ASR implementation through the stable stage slot."""

    if context.request.status_callback is not None:
        context.request.status_callback("transcribing")

    result = transcribe_audio(
        context.pipeline,
        context.working_audio_path,
        language=context.request.language,
        no_repeat_ngram_size=context.request.no_repeat_ngram_size,
        provider_name=context.request.provider_for("asr"),
    )
    context.transcription_result = result.transcription_result
    context.metadata["asr"] = {
        "segment_count": len(result.transcription_result.get("segments", [])),
        "language": result.transcription_result.get("language"),
    }
