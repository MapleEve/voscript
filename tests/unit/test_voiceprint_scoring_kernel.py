"""Golden tests for voiceprint scoring oracle and Rust bridge contracts."""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from providers.kernel_bridge import RustKernelBridgeError, voiceprint_score
from voiceprints.scoring import (
    VoiceprintScoreCandidate,
    score_voiceprint_candidates,
)


def _vec(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)


def _cohort(angles: list[float]) -> np.ndarray:
    return np.stack([_vec(angle) for angle in angles], axis=0)


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
    response = {
        "matched_id": "spk_alice",
        "matched_name": "Alice",
        "similarity": 0.72,
        "reason": "matched",
        "asnorm_active": False,
        "asnorm_reason": "not_requested",
        "candidates": [],
    }

    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(voiceprint_score=lambda payload: response)

    assert (
        voiceprint_score({"query_embedding": [1.0, 0.0]}, importer=_importer)
        == response
    )


def test_kernel_bridge_hard_fails_invalid_voiceprint_score_response():
    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(voiceprint_score=lambda payload: {"ok": True})

    with pytest.raises(RustKernelBridgeError, match="missing keys"):
        voiceprint_score({"query_embedding": [1.0, 0.0]}, importer=_importer)


def test_kernel_bridge_hard_fails_invalid_voiceprint_candidate_response():
    response = {
        "matched_id": "spk_alice",
        "matched_name": "Alice",
        "similarity": 0.72,
        "reason": "matched",
        "asnorm_active": False,
        "asnorm_reason": "not_requested",
        "candidates": [{"speaker_id": "spk_alice"}],
    }

    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(voiceprint_score=lambda payload: response)

    with pytest.raises(RustKernelBridgeError, match="candidate.*missing keys"):
        voiceprint_score({"query_embedding": [1.0, 0.0]}, importer=_importer)


def test_kernel_bridge_hard_fails_non_finite_voiceprint_response():
    response = {
        "matched_id": None,
        "matched_name": None,
        "similarity": float("nan"),
        "reason": "below_threshold",
        "asnorm_active": False,
        "asnorm_reason": "not_requested",
        "candidates": [],
    }

    def _importer(module_name):
        assert module_name == "voscript_core"
        return SimpleNamespace(voiceprint_score=lambda payload: response)

    with pytest.raises(RustKernelBridgeError, match="similarity must be finite"):
        voiceprint_score({"query_embedding": [1.0, 0.0]}, importer=_importer)
