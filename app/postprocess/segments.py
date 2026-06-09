"""Pure helpers for result-segment post-processing.

These helpers are the Python oracle for Rust post-processing kernels. Keep them
free of model, filesystem, request, and database state so the cross-language
contract stays small and directly testable.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

MERGE_GAP_SECONDS = 0.05


def _number(value: object, *, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if isfinite(parsed) else default


def _rounded_time(value: object) -> float:
    return round(max(0.0, _number(value)), 3)


def _rounded_score(value: object) -> float:
    return round(_number(value), 4)


def normalize_words(raw_words: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalize model word payloads to JSON-safe plain dictionaries."""

    if not raw_words:
        return []

    normalized: list[dict[str, Any]] = []
    for raw_word in raw_words:
        word = raw_word if isinstance(raw_word, Mapping) else {}
        normalized.append(
            {
                "word": str(word.get("word", "")),
                "start": _rounded_time(word.get("start", 0.0)),
                "end": _rounded_time(word.get("end", 0.0)),
                "score": _rounded_score(word.get("score", 0.0)),
            }
        )
    return normalized


def _normalize_aligned_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    result = {
        "start": _rounded_time(segment.get("start", 0.0)),
        "end": _rounded_time(segment.get("end", 0.0)),
        "text": str(segment.get("text", "")).strip(),
        "speaker": str(segment.get("speaker", "UNKNOWN") or "UNKNOWN"),
    }
    words = normalize_words(segment.get("words"))
    if words:
        result["words"] = words
    return result


def _can_merge_segments(
    previous: Mapping[str, Any], current: Mapping[str, Any]
) -> bool:
    if previous.get("speaker") != current.get("speaker"):
        return False
    if previous.get("words") or current.get("words"):
        return False
    previous_end = _number(previous.get("end", 0.0))
    current_start = _number(current.get("start", 0.0))
    return current_start <= previous_end + MERGE_GAP_SECONDS


def merge_aligned_segments(
    aligned_segments: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge adjacent text-only segments for the same stable speaker label."""

    merged: list[dict[str, Any]] = []
    for raw_segment in aligned_segments:
        segment = _normalize_aligned_segment(raw_segment)
        if merged and _can_merge_segments(merged[-1], segment):
            previous = merged[-1]
            previous["end"] = max(
                _rounded_time(previous.get("end", 0.0)),
                _rounded_time(segment.get("end", 0.0)),
            )
            previous_text = str(previous.get("text", "")).strip()
            current_text = str(segment.get("text", "")).strip()
            previous["text"] = " ".join(
                part for part in (previous_text, current_text) if part
            )
            continue
        merged.append(segment)
    return merged


def build_display_names(
    speaker_labels: list[str],
    speaker_map: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Disambiguate duplicate enrolled display names without merging speakers."""

    labels_by_name: dict[str, list[str]] = {}

    for speaker_label in speaker_labels:
        match = speaker_map.get(speaker_label, {})
        speaker_name = str(match.get("matched_name") or speaker_label)
        labels_by_name.setdefault(speaker_name, []).append(speaker_label)

    display_names: dict[str, str] = {}
    for speaker_name, labels in labels_by_name.items():
        for index, speaker_label in enumerate(labels, start=1):
            display_names[speaker_label] = (
                speaker_name if index == 1 else f"{speaker_name} ({index})"
            )
    return display_names


def build_result_segments(
    aligned_segments: list[dict[str, Any]],
    speaker_map: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build public result segments while preserving stable speaker labels."""

    merged_segments = merge_aligned_segments(aligned_segments)
    speaker_labels = list(
        dict.fromkeys(segment["speaker"] for segment in merged_segments)
    )
    display_names = build_display_names(speaker_labels, speaker_map)
    segments: list[dict[str, Any]] = []
    seen_speakers: set[str] = set()
    unique_speakers: list[str] = []

    for index, segment in enumerate(merged_segments):
        speaker_label = segment["speaker"]
        match = speaker_map.get(speaker_label, {})
        speaker_name = display_names.get(speaker_label, speaker_label)
        output = {
            "id": index,
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"],
            "speaker_label": speaker_label,
            "speaker_id": match.get("matched_id"),
            "speaker_name": speaker_name,
            "similarity": match.get("similarity", 0),
        }
        if segment.get("words"):
            output["words"] = segment["words"]
        segments.append(output)

        if speaker_name not in seen_speakers:
            seen_speakers.add(speaker_name)
            unique_speakers.append(speaker_name)

    return segments, unique_speakers
