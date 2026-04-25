"""Unit tests for ASR repetition hallucination suppression."""

from providers.asr.default import suppress_repetition_hallucinations


def test_suppresses_single_segment_repeated_prompt_phrase():
    segments = [
        {
            "start": 0.0,
            "end": 30.0,
            "text": "请以简体中文输出，请以简体中文输出。请以简体中文输出。",
        }
    ]

    filtered, report = suppress_repetition_hallucinations(segments)

    assert filtered == []
    assert report["removed_segment_count"] == 1
    assert report["removed_duration"] == 30.0


def test_suppresses_multi_segment_repeated_prompt_run():
    segments = [
        {"start": 0.0, "end": 30.0, "text": "请以简体中文输出。"},
        {"start": 30.0, "end": 61.0, "text": "请以简体中文输出"},
        {"start": 61.0, "end": 139.0, "text": "请以简体中文输出，请以简体中文输出。"},
    ]

    filtered, report = suppress_repetition_hallucinations(segments)

    assert filtered == []
    assert report["removed_segment_count"] == 3
    assert report["removed_duration"] == 139.0


def test_suppresses_long_repeated_non_prompt_segment():
    segments = [
        {
            "start": 0.0,
            "end": 20.0,
            "text": "谢谢观看谢谢观看谢谢观看谢谢观看谢谢观看",
        }
    ]

    filtered, report = suppress_repetition_hallucinations(segments)

    assert filtered == []
    assert report["removed_segment_count"] == 1
    assert report["removed_duration"] == 20.0


def test_keeps_normal_short_repetition_below_hallucination_thresholds():
    segments = [
        {"start": 0.0, "end": 1.0, "text": "对"},
        {"start": 1.0, "end": 2.0, "text": "对"},
        {"start": 2.0, "end": 3.0, "text": "对"},
        {"start": 3.0, "end": 4.0, "text": "对"},
    ]

    filtered, report = suppress_repetition_hallucinations(segments)

    assert filtered == segments
    assert report["removed_segment_count"] == 0


def test_keeps_chinese_repetition_when_text_has_real_context():
    segments = [
        {
            "start": 0.0,
            "end": 12.0,
            "text": "这个问题很重要，所以我再说一遍，这个问题很重要。",
        },
        {
            "start": 12.0,
            "end": 22.0,
            "text": "后面我们继续看实际的处理方案。",
        },
    ]

    filtered, report = suppress_repetition_hallucinations(segments)

    assert filtered == segments
    assert report["removed_segment_count"] == 0
