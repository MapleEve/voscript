"""Default provider for ASR via pipeline-owned Whisper resources."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from pipeline.contracts import ASRProvider, ASRRequest, ASRResult

logger = logging.getLogger(__name__)

_PROMPT_CONTAMINATION_MARKERS = (
    "请以简体中文输出",
    "简体中文输出",
    "以下是普通话的对话",
)


def _duration(segment: dict[str, Any]) -> float:
    return max(
        0.0,
        float(segment.get("end", 0.0)) - float(segment.get("start", 0.0)),
    )


def _normalize_repetition_text(text: str) -> str:
    return "".join(char.lower() for char in text if char.isalnum())


def _prompt_marker_score(normalized_text: str) -> tuple[int, float]:
    if not normalized_text:
        return 0, 0.0

    best_count = 0
    best_ratio = 0.0
    for marker in _PROMPT_CONTAMINATION_MARKERS:
        count = normalized_text.count(marker)
        ratio = (count * len(marker)) / len(normalized_text)
        if count > best_count or ratio > best_ratio:
            best_count = max(best_count, count)
            best_ratio = max(best_ratio, ratio)
    return best_count, best_ratio


def _prompt_marker_key(normalized_text: str) -> str:
    for marker in _PROMPT_CONTAMINATION_MARKERS:
        if marker in normalized_text:
            return f"prompt:{marker}"
    return ""


def _dominant_repeated_unit(normalized_text: str) -> tuple[str, int, float]:
    """Return the dominant repeated short unit, repeat count, and coverage ratio."""

    if len(normalized_text) < 6:
        return "", 0, 0.0

    best_unit = ""
    best_count = 0
    best_ratio = 0.0
    max_unit_len = min(16, len(normalized_text) // 2)
    for unit_len in range(2, max_unit_len + 1):
        chunks = [
            normalized_text[index : index + unit_len]
            for index in range(0, len(normalized_text), unit_len)
        ]
        full_chunks = [chunk for chunk in chunks if len(chunk) == unit_len]
        if not full_chunks:
            continue
        unit, count = Counter(full_chunks).most_common(1)[0]
        ratio = (count * unit_len) / len(normalized_text)
        if count > best_count or ratio > best_ratio:
            best_unit = unit
            best_count = count
            best_ratio = ratio
    return best_unit, best_count, best_ratio


def _is_single_segment_hallucination(segment: dict[str, Any]) -> bool:
    text = str(segment.get("text", "")).strip()
    normalized = _normalize_repetition_text(text)
    if not normalized:
        return False

    duration = _duration(segment)
    marker_count, marker_ratio = _prompt_marker_score(normalized)
    if duration >= 3.0 and marker_count >= 2 and marker_ratio >= 0.55:
        return True

    unit, repeat_count, repeat_ratio = _dominant_repeated_unit(normalized)
    return (
        bool(unit) and duration >= 12.0 and repeat_count >= 4 and repeat_ratio >= 0.82
    )


def _is_repeated_run_hallucination(run: list[dict[str, Any]]) -> bool:
    if len(run) < 3:
        return False

    total_duration = sum(_duration(segment) for segment in run)
    normalized_text = "".join(
        _normalize_repetition_text(str(segment.get("text", ""))) for segment in run
    )
    marker_count, marker_ratio = _prompt_marker_score(normalized_text)
    if total_duration >= 10.0 and marker_count >= 3 and marker_ratio >= 0.55:
        return True

    unit, repeat_count, repeat_ratio = _dominant_repeated_unit(normalized_text)
    return (
        bool(unit)
        and total_duration >= 20.0
        and repeat_count >= 6
        and repeat_ratio >= 0.88
    )


def suppress_repetition_hallucinations(
    segments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Drop ASR segments that are dominated by prompt-like repeated hallucinations."""

    removed_indices: set[int] = set()
    for index, segment in enumerate(segments):
        if _is_single_segment_hallucination(segment):
            removed_indices.add(index)

    run: list[tuple[int, dict[str, Any]]] = []
    previous_key = ""
    for index, segment in enumerate(segments):
        normalized = _normalize_repetition_text(str(segment.get("text", "")))
        key = _prompt_marker_key(normalized) or normalized
        if key and key == previous_key:
            run.append((index, segment))
        else:
            if _is_repeated_run_hallucination([item[1] for item in run]):
                removed_indices.update(item[0] for item in run)
            run = [(index, segment)] if key else []
            previous_key = key

    if _is_repeated_run_hallucination([item[1] for item in run]):
        removed_indices.update(item[0] for item in run)

    filtered = [
        segment
        for index, segment in enumerate(segments)
        if index not in removed_indices
    ]
    removed_duration = round(
        sum(
            _duration(segment)
            for index, segment in enumerate(segments)
            if index in removed_indices
        ),
        3,
    )
    report = {
        "status": "filtered" if removed_indices else "pass",
        "input_segment_count": len(segments),
        "output_segment_count": len(filtered),
        "removed_segment_count": len(removed_indices),
        "removed_duration": removed_duration,
    }
    return filtered, report


def run_faster_whisper_asr(
    pipeline,
    audio_path: str,
    *,
    language: str | None = None,
    no_repeat_ngram_size: int | None = None,
) -> dict[str, object]:
    """Run faster-whisper through the pipeline resource facade."""

    lang_arg = language if language else None
    initial_prompt = "简体中文普通话对话。" if lang_arg is None else None
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
        condition_on_previous_text=False,
    )
    if no_repeat_ngram_size and no_repeat_ngram_size >= 3:
        whisper_kwargs["no_repeat_ngram_size"] = no_repeat_ngram_size

    segments_iter, info = pipeline.whisper.transcribe(audio_path, **whisper_kwargs)
    raw_segments = [
        {
            "start": round(float(segment.start), 3),
            "end": round(float(segment.end), 3),
            "text": segment.text.strip(),
        }
        for segment in segments_iter
    ]
    segments, hallucination_guard = suppress_repetition_hallucinations(raw_segments)
    detected_language = info.language
    logger.info(
        "Transcription done: %d segments, language=%s, repetition_guard=%s",
        len(segments),
        detected_language,
        hallucination_guard["status"],
    )
    return {
        "segments": segments,
        "language": detected_language,
        "hallucination_guard": hallucination_guard,
    }


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
    "suppress_repetition_hallucinations",
]
