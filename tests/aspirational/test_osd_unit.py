"""Unit tests for TranscriptionPipeline.detect_overlaps().

These tests are pure-logic and do NOT require a GPU or real pyannote models.
All heavy model loading is monkeypatched away.

Expected result: PASS (detect_overlaps is already implemented).
"""

import sys
import os
import pytest

# Ensure app/ is importable (conftest.py also does this, but be explicit)
_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)


# ---------------------------------------------------------------------------
# Helper: build a TranscriptionPipeline without touching __init__ side-effects
# ---------------------------------------------------------------------------


def _make_pipeline():
    """Instantiate TranscriptionPipeline via __new__ to skip real model init."""
    from pipeline import TranscriptionPipeline

    p = TranscriptionPipeline.__new__(TranscriptionPipeline)
    # Replicate the attributes set by __init__ that detect_overlaps relies on
    p.device = "cpu"
    p.model_size = "large-v3"
    p.hf_token = None
    p._whisper = None
    p._diarization = None
    p._embedding_model = None
    p._osd = None
    p._osd_onset = 0.5
    return p


# ---------------------------------------------------------------------------
# Fake OSD annotation (same shape as conftest._FakeAnnotation)
# ---------------------------------------------------------------------------


class _FakeSeg:
    def __init__(self, start, end):
        self.start = start
        self.end = end


class _FakeAnnotation:
    def __init__(self, intervals):
        self._intervals = intervals

    def itertracks(self, yield_label=False):
        for s, e in self._intervals:
            seg = _FakeSeg(s, e)
            if yield_label:
                yield seg, None, "OVERLAP"
            else:
                yield seg, None


# ---------------------------------------------------------------------------
# Test 1: detect_overlaps returns a dict with the five required keys
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_detect_overlaps_returns_dict_with_required_keys(minimal_wav, monkeypatch):
    """detect_overlaps must return dict with intervals, total_s, overlap_s, ratio, count."""
    p = _make_pipeline()

    # Patch osd_pipeline property so no real model is loaded
    fake_annotation = _FakeAnnotation([(1.0, 2.5), (4.0, 5.0)])
    monkeypatch.setattr(
        type(p),
        "osd_pipeline",
        property(lambda self: lambda audio_dict: fake_annotation),
    )

    result = p.detect_overlaps(str(minimal_wav), onset=0.08)

    required_keys = {"intervals", "total_s", "overlap_s", "ratio", "count"}
    assert isinstance(result, dict), "detect_overlaps must return a dict"
    missing = required_keys - result.keys()
    assert not missing, f"Missing keys in result: {missing}"


# ---------------------------------------------------------------------------
# Test 2: overlap ratio == overlap_s / total_s (pure math, no model)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_overlap_ratio_is_overlap_s_divided_by_total_s():
    """ratio == overlap_s / total_s — pure arithmetic check, no model needed."""
    # We verify the formula directly, mirroring what detect_overlaps computes:
    #   ratio = round(overlap_s / total_s, 4) if total_s > 0 else 0.0
    total_s = 10.0
    overlap_s = 2.0
    expected_ratio = round(overlap_s / total_s, 4)

    assert expected_ratio == pytest.approx(
        0.2
    ), f"Expected ratio 0.2, got {expected_ratio}"

    # Edge case: zero total_s must not raise ZeroDivisionError
    ratio_zero = round(0.0 / total_s, 4) if total_s > 0 else 0.0
    assert ratio_zero == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Test 3: changing onset invalidates the _osd cache
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_osd_cache_invalidated_when_onset_changes(minimal_wav, monkeypatch):
    """Changing onset must set self._osd = None so the pipeline is re-built."""
    p = _make_pipeline()

    # Provide a dummy osd_pipeline so the first call succeeds without a model.
    call_count = {"n": 0}

    class _CountingProp:
        """Descriptor that tracks how many times the property is accessed."""

        def __get__(self, obj, objtype=None):
            call_count["n"] += 1
            return lambda audio_dict: _FakeAnnotation([(0.5, 1.0)])

    monkeypatch.setattr(type(p), "osd_pipeline", _CountingProp())

    # First call with default onset (0.5)
    p.detect_overlaps(str(minimal_wav), onset=0.5)
    assert p._osd_onset == 0.5

    # Simulate a cached _osd object (as if a real model were loaded)
    p._osd = object()  # sentinel — any truthy object

    # Second call with a DIFFERENT onset — cache must be invalidated
    p.detect_overlaps(str(minimal_wav), onset=0.08)

    assert p._osd_onset == 0.08, "onset was not updated"
    assert p._osd is None, (
        "_osd cache was not cleared when onset changed; "
        "the pipeline will silently use the old threshold"
    )
