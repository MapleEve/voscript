"""Cohort loading, building and persistence for voiceprint AS-norm."""

from __future__ import annotations

import base64
import glob as _glob
import json
import logging
import time as _time
from pathlib import Path

import numpy as np

from .scoring import ASNormScorer

logger = logging.getLogger(__name__)


class VoiceprintCohortManager:
    """Owns cohort file IO and AS-norm scorer lifecycle for VoiceprintDB."""

    def __init__(self, db, cohort_path: str | Path | None, embedding_dim: int):
        self._db = db
        self._cohort_path = Path(cohort_path) if cohort_path is not None else None
        self._embedding_dim = embedding_dim

    @property
    def cohort_path(self) -> Path | None:
        return self._cohort_path

    @property
    def cohort_size(self) -> int:
        scorer = self._db._asnorm
        if scorer is None:
            return 0
        return scorer.cohort_size

    def resolve_path(
        self,
        transcriptions_dir: str | Path | None = None,
        save_path: str | Path | None = None,
    ) -> Path | None:
        if save_path is not None:
            return Path(save_path)
        if self._cohort_path is not None:
            return self._cohort_path
        if transcriptions_dir is not None:
            return Path(transcriptions_dir) / "asnorm_cohort.npy"
        return None

    def load(self, cohort_path: str, top_n: int = 200):
        arr = np.load(cohort_path, allow_pickle=False).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Cohort must be 2D, got {arr.ndim}")
        self._db._asnorm = ASNormScorer(arr, top_n=top_n)
        logger.info("AS-norm cohort loaded: %d speakers, top_n=%d", len(arr), top_n)

    def build_from_transcriptions(
        self, transcriptions_dir: str, save_path: str | None = None
    ) -> int:
        target_gen = self._db._cohort_generation
        if not self._db._cohort_rebuild_lock.acquire(blocking=False):
            logger.info("build_cohort: rebuild already in progress, skipping")
            return self.cohort_size

        try:
            save_target = self.resolve_path(
                transcriptions_dir=transcriptions_dir, save_path=save_path
            )
            embs = []
            skipped_files = 0
            expected_shape = (self._embedding_dim,)

            for result_path in _glob.glob(
                str(Path(transcriptions_dir) / "*/result.json")
            ):
                try:
                    with open(result_path, encoding="utf-8") as fh:
                        payload = json.load(fh)
                    added_from_json = self._collect_json_embeddings(
                        payload=payload,
                        expected_shape=expected_shape,
                        collected=embs,
                    )
                    if added_from_json == 0:
                        skipped_files += self._collect_npy_embeddings(
                            result_path=Path(result_path),
                            expected_shape=expected_shape,
                            collected=embs,
                        )
                except Exception as exc:
                    skipped_files += 1
                    logger.warning("build_cohort: skip %s: %s", result_path, exc)

            self._db.last_cohort_skipped = skipped_files

            if not embs:
                logger.warning(
                    "build_cohort_from_transcriptions: no embeddings found in %s",
                    transcriptions_dir,
                )
                return 0

            cohort = np.stack(embs, axis=0)
            if save_target is not None:
                save_target.parent.mkdir(parents=True, exist_ok=True)
                np.save(save_target, cohort)
                logger.info(
                    "Cohort saved: %d embeddings → %s", len(cohort), save_target
                )

            new_scorer = ASNormScorer(cohort, top_n=min(200, len(cohort)))
            with self._db._lock:
                self._db._asnorm = new_scorer
                if self._db._cohort_generation == target_gen:
                    self._db._cohort_built_gen = target_gen

            logger.info(
                "AS-norm cohort built from transcriptions: %d embeddings",
                len(cohort),
            )
            return len(cohort)
        finally:
            self._db._cohort_rebuild_lock.release()

    def maybe_rebuild(self, transcriptions_dir: str, debounce_s: float = 30.0) -> bool:
        if self._db._cohort_generation == self._db._cohort_built_gen:
            return False
        if _time.monotonic() - self._db._cohort_last_enroll < debounce_s:
            return False

        try:
            size = self.build_from_transcriptions(transcriptions_dir)
            save_target = self.resolve_path(transcriptions_dir=transcriptions_dir)
            if save_target is not None:
                logger.info(
                    "auto-rebuild: AS-norm cohort updated (%d embeddings) → %s",
                    size,
                    save_target,
                )
            else:
                logger.info(
                    "auto-rebuild: AS-norm cohort updated (%d embeddings)", size
                )
            return size > 0
        except Exception as exc:
            logger.warning("auto-rebuild: cohort rebuild failed: %s", exc)
            return False

    @staticmethod
    def _collect_json_embeddings(
        *,
        payload: dict,
        expected_shape: tuple[int, ...],
        collected: list[np.ndarray],
    ) -> int:
        added = 0
        for value in payload.get("speaker_embeddings", {}).values():
            if isinstance(value, list):
                arr = np.array(value, dtype=np.float32)
            elif isinstance(value, str):
                arr = np.frombuffer(base64.b64decode(value), dtype=np.float32)
            else:
                continue

            if arr.shape == expected_shape:
                collected.append(arr)
                added += 1
        return added

    @staticmethod
    def _collect_npy_embeddings(
        *,
        result_path: Path,
        expected_shape: tuple[int, ...],
        collected: list[np.ndarray],
    ) -> int:
        skipped = 0
        for npy_path in result_path.parent.glob("emb_*.npy"):
            try:
                arr = (
                    np.load(str(npy_path), allow_pickle=False)
                    .flatten()
                    .astype(np.float32)
                )
                if arr.shape == expected_shape:
                    collected.append(arr)
            except Exception as exc:
                skipped += 1
                logger.warning(
                    "build_cohort: skip %s due to load error: %s",
                    npy_path,
                    exc,
                )
        return skipped
