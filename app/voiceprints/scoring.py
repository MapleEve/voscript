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
