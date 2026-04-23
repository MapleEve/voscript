"""Unit tests for pure pipeline diarization-alignment helpers."""

from pipeline.stages.diarization.alignment import (
    assign_segment_speaker,
    build_aligned_segments,
    dedup_short_segments,
    normalize_words,
)


def test_assign_segment_speaker_prefers_max_overlap():
    turns = [
        {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        {"start": 0.5, "end": 2.0, "speaker": "SPEAKER_01"},
    ]

    speaker = assign_segment_speaker(0.6, 1.4, turns)

    assert speaker == "SPEAKER_01"


def test_assign_segment_speaker_falls_back_to_midpoint():
    turns = [{"start": 1.0, "end": 2.0, "speaker": "SPEAKER_02"}]

    speaker = assign_segment_speaker(1.5, 1.5, turns)

    assert speaker == "SPEAKER_02"


def test_normalize_words_returns_json_safe_values():
    words = normalize_words(
        [{"word": 7, "start": "1.2349", "end": 2, "score": "0.98765"}]
    )

    assert words == [
        {"word": "7", "start": 1.235, "end": 2.0, "score": 0.9877}
    ]


def test_build_aligned_segments_attaches_speaker_and_words():
    segments = [
        {
            "start": 0.0,
            "end": 1.2,
            "text": " hello ",
            "words": [{"word": "hi", "start": 0.01, "end": 0.4, "score": 0.5}],
        }
    ]
    turns = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]

    aligned = build_aligned_segments(segments, turns)

    assert aligned == [
        {
            "start": 0.0,
            "end": 1.2,
            "text": "hello",
            "speaker": "SPEAKER_00",
            "words": [
                {"word": "hi", "start": 0.01, "end": 0.4, "score": 0.5}
            ],
        }
    ]


def test_dedup_short_segments_drops_consecutive_short_duplicates():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "嗯", "speaker": "SPEAKER_00"},
        {"start": 1.1, "end": 2.0, "text": "嗯", "speaker": "SPEAKER_00"},
        {"start": 2.1, "end": 4.5, "text": "嗯", "speaker": "SPEAKER_00"},
    ]

    deduped = dedup_short_segments(segments)

    assert deduped == [segments[0], segments[2]]
