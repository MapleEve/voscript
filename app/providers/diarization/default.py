"""Default provider for diarization and forced-alignment handoff."""

from __future__ import annotations

import logging

from pipeline.contracts import (
    DiarizationProvider,
    DiarizationRequest,
    DiarizationResult,
)
from pipeline.stages.diarization.alignment import (
    build_aligned_segments,
    dedup_short_segments,
)

logger = logging.getLogger(__name__)


def run_pyannote_diarization(
    pipeline,
    audio_path: str,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict[str, object]]:
    """Run the current pyannote diarization resource and return speaker turns."""

    kwargs = {}
    if min_speakers:
        kwargs["min_speakers"] = min_speakers
    if max_speakers:
        kwargs["max_speakers"] = max_speakers

    result = pipeline.diarization(audio_path, **kwargs)
    turns: list[dict[str, object]] = []
    for turn, _, speaker in result.itertracks(yield_label=True):
        turns.append(
            {
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            }
        )
    return turns


def align_diarized_segments(
    pipeline,
    transcription_result: dict[str, object],
    diarization_turns: list[dict[str, object]],
    audio_path: str,
) -> list[dict[str, object]]:
    """Align ASR output and attach diarization speaker labels."""

    import whisperx

    segments = transcription_result.get("segments", [])
    language = transcription_result.get("language") or "zh"
    audio = whisperx.load_audio(audio_path)

    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=language,
            device=pipeline.device,
        )
        aligned_result = whisperx.align(
            segments,
            align_model,
            align_metadata,
            audio,
            pipeline.device,
            return_char_alignments=False,
        )
        segments = aligned_result.get("segments", segments)
        logger.info("WhisperX forced alignment succeeded for language=%s", language)
    except Exception as exc:
        logger.warning(
            "WhisperX forced alignment failed for language=%s (%s); "
            "continuing without word-level timestamps.",
            language,
            exc,
        )

    return build_aligned_segments(segments, diarization_turns)


class PipelineMethodDiarizationProvider(DiarizationProvider):
    """Run diarization and alignment through pipeline-owned resources."""

    def diarize(self, request: DiarizationRequest) -> DiarizationResult:
        turns = run_pyannote_diarization(
            request.pipeline,
            request.audio_path,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
        )
        aligned = align_diarized_segments(
            request.pipeline,
            request.transcription_result,
            turns,
            request.audio_path,
        )
        deduped = dedup_short_segments(aligned)
        return DiarizationResult(
            turns=turns,
            aligned_segments=deduped,
            dedup_removed=len(aligned) - len(deduped),
        )


default_diarization_provider = PipelineMethodDiarizationProvider()


__all__ = [
    "PipelineMethodDiarizationProvider",
    "align_diarized_segments",
    "default_diarization_provider",
    "run_pyannote_diarization",
]
