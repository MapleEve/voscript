"""Golden tests for voiceprint scoring oracle and Rust bridge contracts."""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_APP_DIR = Path(__file__).resolve().parents[2] / "app"
sys.path.insert(0, str(_APP_DIR))

from providers.kernel_bridge import RustKernelBridgeError, voiceprint_score  # noqa: E402

_SCORING_SPEC = importlib.util.spec_from_file_location(
    "_voscript_voiceprint_scoring", _APP_DIR / "voiceprints" / "scoring.py"
)
assert _SCORING_SPEC is not None and _SCORING_SPEC.loader is not None
_SCORING = importlib.util.module_from_spec(_SCORING_SPEC)
sys.modules[_SCORING_SPEC.name] = _SCORING
_SCORING_SPEC.loader.exec_module(_SCORING)
VoiceprintScoreCandidate = _SCORING.VoiceprintScoreCandidate
score_voiceprint_candidates = _SCORING.score_voiceprint_candidates


def _vec(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)


def _cohort(angles: list[float]) -> np.ndarray:
    return np.stack([_vec(angle) for angle in angles], axis=0)


def _voiceprint_response(**overrides):
    response = {
        "matched_id": "spk_alice",
        "matched_name": "Alice",
        "similarity": 0.72,
        "reason": "matched",
        "asnorm_active": False,
        "asnorm_reason": "not_requested",
        "candidates": [],
    }
    response.update(overrides)
    return response


def _candidate_response(**overrides):
    candidate = {
        "speaker_id": "spk_alice",
        "name": "Alice",
        "raw_similarity": 0.72,
        "similarity": 0.72,
        "effective_threshold": 0.7,
        "score_method": "raw_cosine",
        "sample_count": 1,
        "sample_spread": None,
    }
    candidate.update(overrides)
    return candidate


def test_python_oracle_matches_raw_top_candidate_with_adaptive_threshold():
    result = score_voiceprint_candidates(
        query_embedding=_vec(0.0),
        candidates=[
            VoiceprintScoreCandidate(
                speaker_id="spk_alice",
                name="Alice",
                embedding=_vec(math.acos(0.72)),
                sample_count=1,
                sample_spread=None,
            ),
            VoiceprintScoreCandidate(
                speaker_id="spk_bob",
                name="Bob",
                embedding=_vec(math.acos(0.69)),
                sample_count=3,
                sample_spread=0.0,
            ),
        ],
        threshold=0.75,
    )

    assert result.matched_id == "spk_alice"
    assert result.matched_name == "Alice"
    assert result.reason == "matched"
    assert result.asnorm_active is False
    assert result.asnorm_reason == "not_requested"
    assert result.similarity == pytest.approx(0.72, abs=1e-6)
    assert [candidate.speaker_id for candidate in result.candidates] == [
        "spk_alice",
        "spk_bob",
    ]
    assert result.candidates[0].effective_threshold == pytest.approx(0.70)
    assert result.candidates[0].score_method == "raw_cosine"


def test_python_oracle_falls_back_to_raw_when_asnorm_cohort_is_too_small():
    result = score_voiceprint_candidates(
        query_embedding=_vec(0.0),
        candidates=[
            VoiceprintScoreCandidate(
                speaker_id="spk_alice",
                name="Alice",
                embedding=_vec(0.0),
                sample_count=1,
                sample_spread=None,
            )
        ],
        threshold=0.75,
        cohort=_cohort([0.0, 0.1, -0.1, 0.2, -0.2]),
    )

    assert result.matched_id == "spk_alice"
    assert result.asnorm_active is False
    assert result.asnorm_reason == "cohort_too_small"
    assert result.candidates[0].score_method == "raw_cosine"
    assert result.similarity == pytest.approx(1.0, abs=1e-6)


def test_python_oracle_rejects_ambiguous_asnorm_margin():
    result = score_voiceprint_candidates(
        query_embedding=_vec(0.0),
        candidates=[
            VoiceprintScoreCandidate(
                speaker_id="spk_first",
                name="First",
                embedding=_vec(0.0),
                sample_count=3,
                sample_spread=0.0,
            ),
            VoiceprintScoreCandidate(
                speaker_id="spk_second",
                name="Second",
                embedding=_vec(0.005),
                sample_count=3,
                sample_spread=0.0,
            ),
        ],
        threshold=0.75,
        asnorm_threshold=0.5,
        cohort=_cohort([1.0, 1.1, 1.2, 1.3, 1.4, -1.0, -1.1, -1.2, -1.3, -1.4]),
    )

    assert result.matched_id is None
    assert result.matched_name is None
    assert result.reason == "ambiguous_margin"
    assert result.asnorm_active is True
    assert result.asnorm_reason == "active"
    assert result.similarity == pytest.approx(4.89135345, abs=1e-6)
    assert result.candidates[0].score_method == "asnorm"
    assert result.candidates[1].similarity == pytest.approx(4.88978820, abs=1e-6)


def test_python_oracle_rejects_non_finite_embeddings():
    with pytest.raises(ValueError, match="finite"):
        score_voiceprint_candidates(
            query_embedding=np.array([1.0, np.nan], dtype=np.float32),
            candidates=[],
        )


def test_kernel_bridge_validates_voiceprint_score_response():
    response = _voiceprint_response(
        candidates=[_candidate_response(sample_spread=0.01)]
    )

    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(voiceprint_score=lambda payload: response)

    result = voiceprint_score({"query_embedding": [1.0, 0.0]}, importer=_importer)

    assert result["matched_id"] == "spk_alice"
    assert result["candidates"][0]["sample_count"] == 1
    assert result["candidates"][0]["sample_spread"] == pytest.approx(0.01)


def test_kernel_bridge_hard_fails_invalid_voiceprint_score_response():
    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(voiceprint_score=lambda payload: {"ok": True})

    with pytest.raises(RustKernelBridgeError, match="missing keys"):
        voiceprint_score({"query_embedding": [1.0, 0.0]}, importer=_importer)


def test_kernel_bridge_hard_fails_voiceprint_score_call_failure():
    def _importer(module_name):
        assert module_name == "voscript_core"

        def _voiceprint_score(payload):
            raise RuntimeError("boom")

        return SimpleNamespace(voiceprint_score=_voiceprint_score)

    with pytest.raises(RustKernelBridgeError, match="voiceprint_score call failed"):
        voiceprint_score({"query_embedding": [1.0, 0.0]}, importer=_importer)


def test_kernel_bridge_hard_fails_invalid_voiceprint_candidate_response():
    response = _voiceprint_response(candidates=[{"speaker_id": "spk_alice"}])

    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(voiceprint_score=lambda payload: response)

    with pytest.raises(RustKernelBridgeError, match="candidate.*missing keys"):
        voiceprint_score({"query_embedding": [1.0, 0.0]}, importer=_importer)


def test_kernel_bridge_hard_fails_non_finite_voiceprint_response():
    response = _voiceprint_response(similarity=float("nan"))

    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(voiceprint_score=lambda payload: response)

    with pytest.raises(RustKernelBridgeError, match="similarity must be finite"):
        voiceprint_score({"query_embedding": [1.0, 0.0]}, importer=_importer)


@pytest.mark.parametrize(
    ("response", "message"),
    [
        ([], "non-mapping"),
        (_voiceprint_response(reason=""), "reason must be non-empty"),
        (_voiceprint_response(asnorm_active="false"), "asnorm_active must be bool"),
        (_voiceprint_response(asnorm_reason=""), "asnorm_reason must be non-empty"),
        (_voiceprint_response(candidates={}), "candidates must be a list"),
        (_voiceprint_response(similarity="not-a-number"), "similarity must be numeric"),
    ],
)
def test_kernel_bridge_hard_fails_invalid_voiceprint_score_responses(response, message):
    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(voiceprint_score=lambda payload: response)

    with pytest.raises(RustKernelBridgeError, match=message):
        voiceprint_score({"query_embedding": [1.0, 0.0]}, importer=_importer)


@pytest.mark.parametrize(
    ("candidate", "message"),
    [
        ([], "candidate returned a non-mapping"),
        (_candidate_response(name=""), "candidate name must be non-empty"),
        (
            _candidate_response(score_method=""),
            "candidate score_method must be non-empty",
        ),
        (_candidate_response(raw_similarity="bad"), "raw_similarity must be numeric"),
        (
            _candidate_response(similarity=float("inf")),
            "candidate similarity must be finite",
        ),
        (
            _candidate_response(effective_threshold="bad"),
            "effective_threshold must be numeric",
        ),
        (_candidate_response(sample_count="bad"), "sample_count must be integer-like"),
        (_candidate_response(sample_count=-1), "sample_count must be non-negative"),
        (_candidate_response(sample_spread="bad"), "sample_spread must be numeric"),
        (
            _candidate_response(sample_spread=float("nan")),
            "sample_spread must be finite",
        ),
    ],
)
def test_kernel_bridge_hard_fails_invalid_voiceprint_candidate_responses(
    candidate, message
):
    response = _voiceprint_response(candidates=[candidate])

    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(voiceprint_score=lambda payload: response)

    with pytest.raises(RustKernelBridgeError, match=message):
        voiceprint_score({"query_embedding": [1.0, 0.0]}, importer=_importer)
