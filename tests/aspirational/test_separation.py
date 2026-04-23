"""RED tests for TranscriptionPipeline.separate_overlaps().

separate_overlaps does NOT exist yet — these tests are expected to FAIL
with AttributeError.  They define the contract for the method to be
implemented in the GREEN phase.

Expected result: ALL FAIL (AttributeError: type object
'TranscriptionPipeline' has no attribute 'separate_overlaps').
"""

import sys
import os
import inspect
import pytest

_APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app"))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)


# ---------------------------------------------------------------------------
# Test 1: the method must exist on the class
# ---------------------------------------------------------------------------


def test_pipeline_has_separate_overlaps_method():
    """TranscriptionPipeline must have a separate_overlaps method.

    RED: fails because separate_overlaps is not yet implemented.
    """
    from pipeline import TranscriptionPipeline

    assert hasattr(TranscriptionPipeline, "separate_overlaps"), (
        "TranscriptionPipeline is missing the 'separate_overlaps' method. "
        "Implement it in app/pipeline.py."
    )


# ---------------------------------------------------------------------------
# Test 2: the method must accept audio_path and n_speakers parameters
# ---------------------------------------------------------------------------


def test_separate_overlaps_signature():
    """separate_overlaps(audio_path, n_speakers=2) must be callable with those params.

    RED: fails because separate_overlaps does not exist yet.
    """
    from pipeline import TranscriptionPipeline

    sig = inspect.signature(TranscriptionPipeline.separate_overlaps)
    param_names = list(sig.parameters.keys())

    assert "audio_path" in param_names, (
        f"separate_overlaps is missing 'audio_path' parameter. "
        f"Got parameters: {param_names}"
    )
    assert "n_speakers" in param_names, (
        f"separate_overlaps is missing 'n_speakers' parameter. "
        f"Got parameters: {param_names}"
    )


# ---------------------------------------------------------------------------
# Test 3: the method must return a list of str paths, one per speaker
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_separate_overlaps_returns_list_of_audio_paths(minimal_wav, monkeypatch):
    """separate_overlaps must return a list of file paths (str), one per speaker.

    Marked integration because it would ultimately exercise audio I/O, but in
    the RED phase it fails earlier — AttributeError because the method does
    not exist.

    RED: fails with AttributeError.
    """
    from pipeline import TranscriptionPipeline

    # Instantiate without triggering real model loading
    p = TranscriptionPipeline.__new__(TranscriptionPipeline)
    p.device = "cpu"
    p.model_size = "large-v3"
    p.hf_token = None
    p._whisper = None
    p._diarization = None
    p._embedding_model = None
    p._osd = None
    p._osd_onset = 0.5

    result = p.separate_overlaps(str(minimal_wav), n_speakers=2)

    assert isinstance(
        result, list
    ), f"separate_overlaps must return a list, got {type(result)}"
    assert len(result) == 2, f"Expected 2 paths (one per speaker), got {len(result)}"
    assert all(
        isinstance(path, str) for path in result
    ), "All entries in the returned list must be str file paths"
