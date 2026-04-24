"""Pure helpers for transcription/diarization alignment post-processing."""

from __future__ import annotations

from typing import Any


def assign_segment_speaker(
    seg_start: float, seg_end: float, diarization_turns: list[dict[str, Any]]
) -> str:
    """Pick the diarization speaker with the greatest overlap for a segment."""
    seg_mid = (seg_start + seg_end) / 2
    best_speaker = "UNKNOWN"
    best_overlap = 0.0

    for turn in diarization_turns:
        overlap_start = max(seg_start, turn["start"])
        overlap_end = min(seg_end, turn["end"])
        overlap = max(0.0, overlap_end - overlap_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn["speaker"]

    if best_speaker != "UNKNOWN":
        return best_speaker

    for turn in diarization_turns:
        if turn["start"] <= seg_mid <= turn["end"]:
            return turn["speaker"]

    return best_speaker


def normalize_words(raw_words: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Normalise WhisperX word payloads to JSON-safe plain Python dicts."""
    if not raw_words:
        return []

    return [
        {
            "word": str(word.get("word", "")),
            "start": round(float(word.get("start", 0.0)), 3),
            "end": round(float(word.get("end", 0.0)), 3),
            "score": round(float(word.get("score", 0.0)), 4),
        }
        for word in raw_words
    ]


def normalize_segment(
    segment: dict[str, Any], diarization_turns: list[dict[str, Any]]
) -> dict[str, Any]:
    """Attach a speaker label and normalise optional word timings."""
    seg_start = float(segment.get("start", 0.0))
    seg_end = float(segment.get("end", 0.0))
    result = {
        "start": round(seg_start, 3),
        "end": round(seg_end, 3),
        "text": segment.get("text", "").strip(),
        "speaker": assign_segment_speaker(seg_start, seg_end, diarization_turns),
    }

    words = normalize_words(segment.get("words"))
    if words:
        result["words"] = words

    return result


def build_aligned_segments(
    segments: list[dict[str, Any]], diarization_turns: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Normalise aligned segments and assign speakers."""
    return [normalize_segment(segment, diarization_turns) for segment in segments]


def dedup_short_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop consecutive duplicate short segments (backchannel suppression)."""
    if not segments:
        return segments

    result = [segments[0]]
    for segment in segments[1:]:
        previous = result[-1]
        text = segment.get("text", "").strip()
        previous_text = previous.get("text", "").strip()
        duration = segment.get("end", 0.0) - segment.get("start", 0.0)
        if text and text == previous_text and duration < 2.0 and len(text) <= 4:
            continue
        result.append(segment)
    return result
