"""Stable slot for automatic speech recognition."""

from __future__ import annotations

from typing import TYPE_CHECKING

from pipeline.contracts import ASRRequest
from pipeline.registry import resolve_provider

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Run the current ASR implementation through the stable stage slot."""

    if context.request.status_callback is not None:
        context.request.status_callback("transcribing")

    provider = resolve_provider("asr", context.request.provider_for("asr"))
    result = provider.transcribe(
        ASRRequest(
            pipeline=context.pipeline,
            audio_path=context.working_audio_path,
            language=context.request.language,
            no_repeat_ngram_size=context.request.no_repeat_ngram_size,
        )
    )
    context.transcription_result = result.transcription_result
    context.metadata["asr"] = {
        "segment_count": len(result.transcription_result.get("segments", [])),
        "language": result.transcription_result.get("language"),
    }
    hallucination_guard = result.transcription_result.get("hallucination_guard")
    if hallucination_guard is not None:
        context.metadata["asr"]["hallucination_guard"] = hallucination_guard
