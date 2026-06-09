"""Golden tests for pure transcript post-processing kernels."""

from __future__ import annotations

from math import nan

from postprocess.segments import (
    build_display_names,
    build_result_segments,
    merge_aligned_segments,
    normalize_words,
)


def test_normalize_words_handles_missing_none_nan_and_negative_values():
    words = normalize_words(
        [
            {
                "word": None,
                "start": -1,
                "end": nan,
                "score": "not-a-number",
            },
            {},
        ]
    )

    assert words == [
        {"word": "None", "start": 0.0, "end": 0.0, "score": 0.0},
        {"word": "", "start": 0.0, "end": 0.0, "score": 0.0},
    ]


def test_merge_aligned_segments_merges_adjacent_text_only_same_speaker():
    segments = merge_aligned_segments(
        [
            {
                "start": "0",
                "end": 1.0,
                "text": " first ",
                "speaker": "SPEAKER_00",
            },
            {
                "start": 1.03,
                "end": 2.0,
                "text": "second",
                "speaker": "SPEAKER_00",
            },
            {
                "start": 2.0,
                "end": 3.0,
                "text": "third",
                "speaker": "SPEAKER_01",
            },
        ]
    )

    assert segments == [
        {
            "start": 0.0,
            "end": 2.0,
            "text": "first second",
            "speaker": "SPEAKER_00",
        },
        {
            "start": 2.0,
            "end": 3.0,
            "text": "third",
            "speaker": "SPEAKER_01",
        },
    ]


def test_merge_aligned_segments_does_not_merge_word_payloads():
    segments = merge_aligned_segments(
        [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "first",
                "speaker": "SPEAKER_00",
                "words": [{"word": "first", "start": 0, "end": 1, "score": 0.8}],
            },
            {
                "start": 1.0,
                "end": 2.0,
                "text": "second",
                "speaker": "SPEAKER_00",
            },
        ]
    )

    assert len(segments) == 2
    assert segments[0]["speaker"] == "SPEAKER_00"
    assert segments[0]["words"] == [
        {"word": "first", "start": 0.0, "end": 1.0, "score": 0.8}
    ]


def test_display_names_disambiguate_without_rewriting_speaker_labels():
    display_names = build_display_names(
        ["SPEAKER_00", "SPEAKER_01"],
        {
            "SPEAKER_00": {"matched_name": "Maple"},
            "SPEAKER_01": {"matched_name": "Maple"},
        },
    )

    assert display_names == {
        "SPEAKER_00": "Maple",
        "SPEAKER_01": "Maple (2)",
    }


def test_build_result_segments_preserves_raw_label_and_unique_display_names():
    segments, unique_speakers = build_result_segments(
        [
            {"start": 0.0, "end": 1.0, "text": "a", "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "text": "b", "speaker": "SPEAKER_01"},
        ],
        {
            "SPEAKER_00": {
                "matched_id": "spk_same",
                "matched_name": "Maple",
                "similarity": 2.0,
            },
            "SPEAKER_01": {
                "matched_id": "spk_same",
                "matched_name": "Maple",
                "similarity": 1.0,
            },
        },
    )

    assert [segment["speaker_label"] for segment in segments] == [
        "SPEAKER_00",
        "SPEAKER_01",
    ]
    assert [segment["speaker_name"] for segment in segments] == [
        "Maple",
        "Maple (2)",
    ]
    assert unique_speakers == ["Maple", "Maple (2)"]
