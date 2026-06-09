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
ASNORM_MIN_COHORT_SIZE = 10
_MIN_ASNORM_COHORT_SIZE = ASNORM_MIN_COHORT_SIZE

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


@dataclass(frozen=True)
class VoiceprintScoreCandidate:
    speaker_id: str
    name: str
    embedding: np.ndarray
    sample_count: int
    sample_spread: float | None


@dataclass(frozen=True)
class VoiceprintScoredCandidate:
    speaker_id: str
    name: str
    raw_similarity: float
    similarity: float
    effective_threshold: float
    score_method: str
    sample_count: int
    sample_spread: float | None


@dataclass(frozen=True)
class VoiceprintScoreDecision:
    matched_id: str | None
    matched_name: str | None
    similarity: float
    reason: str
    asnorm_active: bool
    asnorm_reason: str
    candidates: tuple[VoiceprintScoredCandidate, ...]


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


def score_voiceprint_candidates(
    *,
    query_embedding: np.ndarray,
    candidates: list[VoiceprintScoreCandidate],
    threshold: float = 0.75,
    asnorm_threshold: float = 0.5,
    cohort: np.ndarray | None = None,
    asnorm_top_n: int = 200,
    asnorm_min_margin: float = _ASNORM_MIN_TOP2_MARGIN,
) -> VoiceprintScoreDecision:
    """Score voiceprint candidates with the Python oracle contract.

    This is the golden oracle for the Rust voiceprint kernel. It owns the
    behavior contract; Rust must match it when selected.
    """

    query = _validated_embedding("query_embedding", query_embedding)
    if not np.isfinite([threshold, asnorm_threshold, asnorm_min_margin]).all():
        raise ValueError("voiceprint thresholds must be finite")
    if not candidates:
        return _voiceprint_no_match(
            reason="no_candidates",
            asnorm_reason="not_requested",
            asnorm_active=False,
            candidates=(),
            similarity=0.0,
        )

    query_norm = float(np.linalg.norm(query))
    if query_norm < 1e-12:
        return _voiceprint_no_match(
            reason="invalid_query",
            asnorm_reason="not_requested",
            asnorm_active=False,
            candidates=(),
            similarity=0.0,
        )

    asnorm_active = False
    asnorm_reason = "not_requested"
    scorer: ASNormScorer | None = None
    if cohort is not None:
        cohort_array = _validated_cohort(cohort, dim=len(query))
        if len(cohort_array) < ASNORM_MIN_COHORT_SIZE:
            asnorm_reason = "cohort_too_small"
        else:
            scorer = ASNormScorer(cohort_array, top_n=asnorm_top_n)
            asnorm_active = True
            asnorm_reason = "active"

    scored_candidates: list[VoiceprintScoredCandidate] = []
    query_normed = query / query_norm
    for candidate in candidates:
        enroll = _validated_embedding("candidate embedding", candidate.embedding)
        if len(enroll) != len(query):
            raise ValueError("voiceprint embeddings must share dimension")
        if candidate.sample_spread is not None and not np.isfinite(
            candidate.sample_spread
        ):
            raise ValueError("voiceprint sample_spread values must be finite")

        enroll_norm = float(np.linalg.norm(enroll))
        raw_similarity = (
            0.0 if enroll_norm < 1e-12 else float((enroll / enroll_norm) @ query_normed)
        )
        if scorer is not None:
            similarity = scorer.score(enroll, query)
            effective = effective_asnorm_threshold(
                base=asnorm_threshold,
                sample_count=candidate.sample_count,
                sample_spread=candidate.sample_spread,
            )
            score_method = "asnorm"
        else:
            similarity = raw_similarity
            effective = effective_threshold(
                base=threshold,
                sample_count=candidate.sample_count,
                sample_spread=candidate.sample_spread,
            )
            score_method = "raw_cosine"

        scored_candidates.append(
            VoiceprintScoredCandidate(
                speaker_id=candidate.speaker_id,
                name=candidate.name,
                raw_similarity=raw_similarity,
                similarity=similarity,
                effective_threshold=effective,
                score_method=score_method,
                sample_count=candidate.sample_count,
                sample_spread=candidate.sample_spread,
            )
        )

    scored_candidates.sort(key=lambda candidate: candidate.similarity, reverse=True)
    scored = tuple(scored_candidates)
    if not scored:
        return _voiceprint_no_match(
            reason="no_candidates",
            asnorm_reason=asnorm_reason,
            asnorm_active=asnorm_active,
            candidates=scored,
            similarity=0.0,
        )

    best = scored[0]
    if asnorm_active:
        second = scored[1].similarity if len(scored) > 1 else None
        if not asnorm_margin_passes(
            best_score=best.similarity,
            second_score=second,
            min_margin=asnorm_min_margin,
        ):
            return _voiceprint_no_match(
                reason="ambiguous_margin",
                asnorm_reason=asnorm_reason,
                asnorm_active=True,
                candidates=scored,
                similarity=best.similarity,
            )

    if best.similarity >= best.effective_threshold:
        return VoiceprintScoreDecision(
            matched_id=best.speaker_id,
            matched_name=best.name,
            similarity=best.similarity,
            reason="matched",
            asnorm_active=asnorm_active,
            asnorm_reason=asnorm_reason,
            candidates=scored,
        )

    return _voiceprint_no_match(
        reason="below_threshold",
        asnorm_reason=asnorm_reason,
        asnorm_active=asnorm_active,
        candidates=scored,
        similarity=best.similarity,
    )


def _voiceprint_no_match(
    *,
    reason: str,
    asnorm_reason: str,
    asnorm_active: bool,
    candidates: tuple[VoiceprintScoredCandidate, ...],
    similarity: float,
) -> VoiceprintScoreDecision:
    return VoiceprintScoreDecision(
        matched_id=None,
        matched_name=None,
        similarity=similarity,
        reason=reason,
        asnorm_active=asnorm_active,
        asnorm_reason=asnorm_reason,
        candidates=candidates,
    )


def _validated_embedding(name: str, embedding: np.ndarray) -> np.ndarray:
    array = np.asarray(embedding, dtype=np.float32).flatten()
    if len(array) == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} values must be finite")
    return array


def _validated_cohort(cohort: np.ndarray, dim: int) -> np.ndarray:
    array = np.asarray(cohort, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("voiceprint cohort must be a 2-D array")
    if array.shape[1] != dim:
        raise ValueError("voiceprint cohort embeddings must share dimension")
    if not np.isfinite(array).all():
        raise ValueError("voiceprint cohort values must be finite")
    return array
