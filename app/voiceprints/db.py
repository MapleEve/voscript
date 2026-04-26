"""VoiceprintDB facade: storage/bootstrap + repository helpers live elsewhere."""

import logging
import os
import threading
import time as _time
from pathlib import Path

import numpy as np

from config import EMBEDDING_DIM
from .cohort import VoiceprintCohortManager
from .repository import VoiceprintRepository
from .scoring import (
    ASNormScorer,
    asnorm_margin_passes,
    effective_asnorm_threshold,
    effective_threshold,
    resolve_score,
)
from .storage import VoiceprintStorage

logger = logging.getLogger(__name__)


class VoiceprintDB:
    """Thread-safe speaker database facade with unchanged public API."""

    def __init__(
        self,
        db_dir: str = "/data/voiceprints",
        cohort_path: str | os.PathLike | None = None,
    ):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._cohort_path = Path(cohort_path) if cohort_path is not None else None

        self._lock = threading.RLock()
        self._asnorm: ASNormScorer | None = None
        self._asnorm_threshold: float = 0.5
        self._cohort_generation: int = 0
        self._cohort_built_gen: int = -1
        self._cohort_last_enroll: float = 0.0
        self._cohort_rebuild_lock = threading.RLock()
        self.last_cohort_skipped: int = 0

        self._storage = VoiceprintStorage(self.db_dir, self._lock)
        self._storage.bootstrap()
        self._repository = VoiceprintRepository(
            storage=self._storage,
            lock=self._lock,
            on_enroll=self._mark_enrolled,
        )
        self._conn = self._storage.conn

        self._cohort_manager = VoiceprintCohortManager(
            self,
            cohort_path=self._cohort_path,
            embedding_dim=EMBEDDING_DIM,
        )

    @property
    def cohort_size(self) -> int:
        return self._cohort_manager.cohort_size

    @property
    def cohort_path(self) -> Path | None:
        return self._cohort_manager.cohort_path

    @property
    def _vec_loaded(self) -> bool:
        return self._storage.vec_loaded

    @property
    def _vec_table_dim(self) -> int | None:
        return self._storage.vec_table_dim

    def _mark_enrolled(self):
        self._cohort_generation += 1
        self._cohort_last_enroll = _time.monotonic()

    def add_speaker(self, name: str, embedding: np.ndarray) -> str:
        return self._repository.add_speaker(name, embedding)

    def update_speaker(
        self, speaker_id: str, new_embedding: np.ndarray, name: str | None = None
    ):
        self._repository.update_speaker(speaker_id, new_embedding, name=name)

    def delete_speaker(self, speaker_id: str):
        self._repository.delete_speaker(speaker_id)

    def rename_speaker(self, speaker_id: str, new_name: str):
        self._repository.rename_speaker(speaker_id, new_name)

    def identify(
        self, embedding: np.ndarray, threshold: float = 0.75
    ) -> tuple[str | None, str | None, float]:
        """Return ``(speaker_id, speaker_name, similarity)`` for the closest match.

        The ``threshold`` argument is a **base** threshold. Per-candidate, we
        compute an *effective* threshold that loosens for noisy clusters:

        - ``sample_count == 1``: the averaged embedding is the single enroll
          sample and has no spread estimate. Loosen the threshold by
          ``_SINGLE_SAMPLE_RELAXATION`` (0.05 by default) to avoid the
          "one-sample enrollment never matches anyone across sessions" failure
          mode.
        - ``sample_count >= 2``: loosen by ``k * sample_spread`` clamped at
          ``_SPREAD_RELAXATION_CAP``. High intra-cluster variance → the speaker
          sounds different across sessions → be more lenient. Low variance → keep
          the strict base threshold.
        - Never drop below ``_ABSOLUTE_FLOOR`` (0.60) regardless of relaxation.

        In AS-norm mode, the effective threshold is a z-score threshold adjusted
        by sample count/spread, and automatic naming also requires top-1/top-2
        separation. Returns ``(None, None, best_similarity)`` when the best
        candidate is below threshold or too ambiguous; ``best_similarity`` uses
        the active scoring method even when rejected.
        """
        query = embedding.flatten().astype(np.float32)
        if float(np.linalg.norm(query)) < 1e-6:
            return None, None, 0.0

        candidates = self._repository.fetch_identify_candidates(query, limit=2)
        if not candidates:
            return None, None, 0.0
        candidate = candidates[0]

        best_sim = candidate.similarity
        score_result = resolve_score(
            raw_similarity=best_sim,
            scorer=self._asnorm,
            enroll_emb=candidate.enroll_emb,
            test_emb=query,
        )
        best_sim = score_result.similarity

        if score_result.asnorm_active:
            candidate_count = self._repository.count_identify_candidates()
            candidates = self._repository.fetch_identify_candidates(
                query, limit=candidate_count
            )
            scored_candidates = []
            for candidate in candidates:
                candidate_score = resolve_score(
                    raw_similarity=candidate.similarity,
                    scorer=self._asnorm,
                    enroll_emb=candidate.enroll_emb,
                    test_emb=query,
                )
                if not candidate_score.asnorm_active:
                    continue
                effective = effective_asnorm_threshold(
                    base=self._asnorm_threshold,
                    sample_count=candidate.sample_count,
                    sample_spread=candidate.sample_spread,
                )
                scored_candidates.append(
                    (candidate, candidate_score.similarity, effective)
                )

            if not scored_candidates:
                return None, None, 0.0

            scored_candidates.sort(key=lambda item: item[1], reverse=True)
            candidate, best_sim, effective = scored_candidates[0]
            second_sim = scored_candidates[1][1] if len(scored_candidates) > 1 else None
            logger.debug(
                "identify[asnorm]: best=%s normalized_sim=%.4f threshold=%.3f "
                "second=%.4f margin_ok=%s (n=%d, spread=%s, candidates=%d)",
                candidate.speaker_id,
                best_sim,
                effective,
                second_sim if second_sim is not None else float("nan"),
                asnorm_margin_passes(best_sim, second_sim),
                candidate.sample_count,
                candidate.sample_spread,
                len(scored_candidates),
            )
            if not asnorm_margin_passes(best_sim, second_sim):
                return None, None, best_sim
        else:
            effective = effective_threshold(
                base=threshold,
                sample_count=candidate.sample_count,
                sample_spread=candidate.sample_spread,
            )
            logger.debug(
                "identify: best=%s best_sim=%.4f base=%.3f effective=%.3f "
                "(n=%d, spread=%s)",
                candidate.speaker_id,
                best_sim,
                threshold,
                effective,
                candidate.sample_count,
                candidate.sample_spread,
            )

        if best_sim >= effective:
            return candidate.speaker_id, candidate.name, best_sim
        return None, None, best_sim

    @staticmethod
    def _effective_threshold(
        base: float, sample_count: int, sample_spread: float | None
    ) -> float:
        return effective_threshold(base, sample_count, sample_spread)

    @staticmethod
    def _effective_asnorm_threshold(
        base: float, sample_count: int, sample_spread: float | None
    ) -> float:
        return effective_asnorm_threshold(base, sample_count, sample_spread)

    def list_speakers(self) -> list[dict]:
        return self._repository.list_speakers()

    def get_speaker(self, speaker_id: str) -> dict | None:
        return self._repository.get_speaker(speaker_id)

    def load_cohort(self, cohort_path: str, top_n: int = 200):
        self._cohort_manager.load(cohort_path, top_n=top_n)

    def build_cohort_from_transcriptions(
        self, transcriptions_dir: str, save_path: str | None = None
    ) -> int:
        return self._cohort_manager.build_from_transcriptions(
            transcriptions_dir, save_path=save_path
        )

    def set_asnorm_threshold(self, threshold: float):
        self._asnorm_threshold = threshold

    def maybe_rebuild_cohort(
        self, transcriptions_dir: str, debounce_s: float = 30.0
    ) -> bool:
        return self._cohort_manager.maybe_rebuild(
            transcriptions_dir, debounce_s=debounce_s
        )
