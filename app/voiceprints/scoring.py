"""Voiceprint scoring helpers and AS-norm primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Threshold tuning knobs (adaptive identification).
# A freshly enrolled speaker has just one sample, so its averaged embedding is
# the single noisy sample — we loosen the match threshold by this much to
# accept the inevitable cross-session drift.
_SINGLE_SAMPLE_RELAXATION = 0.05  # 0.75 - 0.05 = 0.70 by default
# For multi-sample speakers we compute the std of cos(sample_i, avg). The
# dynamic threshold relaxes by k * std, capped so a pathologically noisy
# cluster can't pull the threshold arbitrarily low.
_SPREAD_RELAXATION_K = 3.0
_SPREAD_RELAXATION_CAP = 0.10
# Absolute floor — never accept a match below this, regardless of per-speaker
# relaxation. Guards against false positives from degenerate clusters.
_ABSOLUTE_FLOOR = 0.60
_MIN_ASNORM_COHORT_SIZE = 10

# AS-norm scores are z-score-like, not raw cosine similarities. Keep the
# operating point near the calibrated base for stable multi-sample speakers, but
# require stronger evidence before auto-naming sparse or noisy enrollments.
_ASNORM_SINGLE_SAMPLE_PENALTY = 0.10
_ASNORM_LEGACY_SPREAD_UNKNOWN_PENALTY = 0.05
_ASNORM_LOW_SAMPLE_PENALTY = 0.025
_ASNORM_SPREAD_PENALTY_K = 0.50
_ASNORM_SPREAD_PENALTY_CAP = 0.10
_ASNORM_STABLE_RELAXATION = 0.02
_ASNORM_MIN_TOP2_MARGIN = 0.05


class ASNormScorer:
    """AS-norm score normalization using a cohort of impostor embeddings."""

    def __init__(self, cohort: np.ndarray, top_n: int = 200):
        norms = np.linalg.norm(cohort, axis=1, keepdims=True)
        self._cohort = cohort / (norms + 1e-8)  # (N, 256), L2-normed
        self._top_n = min(top_n, len(cohort))

    @property
    def cohort_size(self) -> int:
        return len(self._cohort)

    @staticmethod
    def _l2(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / (n + 1e-8)

    def _cohort_stats(self, emb: np.ndarray) -> tuple[float, float]:
        scores = self._cohort @ self._l2(emb)
        top = np.sort(scores)[::-1][: self._top_n]
        return float(top.mean()), float(top.std() + 1e-8)

    def score(self, enroll_emb: np.ndarray, test_emb: np.ndarray) -> float:
        raw = float(self._l2(enroll_emb) @ self._l2(test_emb))
        if self.cohort_size < _MIN_ASNORM_COHORT_SIZE:
            return raw
        mean_e, std_e = self._cohort_stats(enroll_emb)
        mean_t, std_t = self._cohort_stats(test_emb)
        return 0.5 * ((raw - mean_e) / std_e + (raw - mean_t) / std_t)


@dataclass(frozen=True)
class ScoreResult:
    similarity: float
    asnorm_active: bool


def resolve_score(
    *,
    raw_similarity: float,
    scorer: ASNormScorer | None,
    enroll_emb: np.ndarray | None,
    test_emb: np.ndarray,
) -> ScoreResult:
    """Return the similarity score that identify() should apply downstream."""
    if scorer is None or enroll_emb is None:
        return ScoreResult(similarity=raw_similarity, asnorm_active=False)

    normalized = scorer.score(enroll_emb, test_emb)
    if scorer.cohort_size < _MIN_ASNORM_COHORT_SIZE:
        return ScoreResult(similarity=raw_similarity, asnorm_active=False)

    return ScoreResult(similarity=normalized, asnorm_active=True)


def effective_threshold(
    base: float, sample_count: int, sample_spread: float | None
) -> float:
    """Adaptive threshold per-candidate."""
    if sample_count <= 1 or sample_spread is None:
        if sample_count <= 1:
            dyn = base - _SINGLE_SAMPLE_RELAXATION
        else:
            dyn = base
    else:
        relax = min(_SPREAD_RELAXATION_K * float(sample_spread), _SPREAD_RELAXATION_CAP)
        dyn = base - relax
    return max(_ABSOLUTE_FLOOR, min(base, dyn))


def effective_asnorm_threshold(
    base: float, sample_count: int, sample_spread: float | None
) -> float:
    """Sample-count-aware threshold for AS-norm z-scores.

    AS-norm uses a different score scale from raw cosine, so this intentionally
    does not reuse the raw cosine relaxation constants. Sparse enrollments need
    a higher score to auto-name; stable multi-sample enrollments stay near the
    AS-norm operating point.
    """
    if sample_count <= 1:
        return base + _ASNORM_SINGLE_SAMPLE_PENALTY

    if sample_spread is None:
        return base + _ASNORM_LEGACY_SPREAD_UNKNOWN_PENALTY

    low_sample_penalty = max(0, 3 - sample_count) * _ASNORM_LOW_SAMPLE_PENALTY
    spread_penalty = min(
        max(0.0, float(sample_spread)) * _ASNORM_SPREAD_PENALTY_K,
        _ASNORM_SPREAD_PENALTY_CAP,
    )
    threshold = base + low_sample_penalty + spread_penalty

    if sample_count >= 3 and float(sample_spread) <= 0.03:
        threshold -= _ASNORM_STABLE_RELAXATION

    return max(0.0, threshold)


def asnorm_margin_passes(
    best_score: float,
    second_score: float | None,
    min_margin: float = _ASNORM_MIN_TOP2_MARGIN,
) -> bool:
    """Return whether top-1 is sufficiently separated from top-2."""
    if second_score is None:
        return True
    return (best_score - second_score) >= min_margin
