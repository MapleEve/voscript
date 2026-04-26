"""Repository helpers for voiceprint speaker reads and writes."""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .storage import VoiceprintStorage, blob_to_emb, emb_to_blob, transaction


@dataclass(frozen=True)
class IdentifyCandidate:
    speaker_id: str
    name: str
    sample_count: int
    sample_spread: float | None
    similarity: float
    enroll_emb: np.ndarray | None


class VoiceprintRepository:
    """Owns transaction-scoped speaker CRUD and read-model queries."""

    def __init__(self, storage: VoiceprintStorage, lock, on_enroll):
        self._storage = storage
        self._conn = storage.conn
        self._lock = lock
        self._on_enroll = on_enroll

    def add_speaker(self, name: str, embedding: np.ndarray) -> str:
        emb = embedding.flatten().astype(np.float32)
        now = datetime.now().isoformat()

        with self._lock:
            existing = self._conn.execute(
                "SELECT id FROM speakers WHERE LOWER(name) = LOWER(?)",
                (name,),
            ).fetchone()
            if existing:
                speaker_id = existing[0]
                self.update_speaker(speaker_id, embedding)
                return speaker_id

            speaker_id = f"spk_{uuid.uuid4().hex[:8]}"
            if self._storage.vec_loaded:
                self._storage.ensure_vec_table(len(emb))

            with transaction(self._conn):
                self._conn.execute(
                    "INSERT INTO speakers(id, name, sample_count, sample_spread, "
                    "created_at, updated_at) VALUES (?, ?, 1, NULL, ?, ?)",
                    (speaker_id, name, now, now),
                )
                self._conn.execute(
                    "INSERT INTO speaker_samples(speaker_id, embedding) VALUES (?, ?)",
                    (speaker_id, emb_to_blob(emb)),
                )
                self._conn.execute(
                    "INSERT INTO speaker_avg(speaker_id, embedding) VALUES (?, ?)",
                    (speaker_id, emb_to_blob(emb)),
                )
                if self._storage.vec_loaded:
                    self._storage.upsert_vec(speaker_id, emb)

            self._on_enroll()
            return speaker_id

    def update_speaker(
        self, speaker_id: str, new_embedding: np.ndarray, name: str | None = None
    ):
        emb = new_embedding.flatten().astype(np.float32)

        with self._lock:
            self._require_speaker(speaker_id)
            with transaction(self._conn):
                self._conn.execute(
                    "INSERT INTO speaker_samples(speaker_id, embedding) VALUES (?, ?)",
                    (speaker_id, emb_to_blob(emb)),
                )
                avg_emb, spread = self._recompute_avg_and_spread(speaker_id)
                self._conn.execute(
                    "INSERT OR REPLACE INTO speaker_avg(speaker_id, embedding) VALUES (?, ?)",
                    (speaker_id, emb_to_blob(avg_emb)),
                )
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM speaker_samples WHERE speaker_id = ?",
                    (speaker_id,),
                ).fetchone()[0]
                now_iso = datetime.now().isoformat()
                if name is not None:
                    self._conn.execute(
                        "UPDATE speakers SET sample_count = ?, sample_spread = ?, "
                        "updated_at = ?, name = ? WHERE id = ?",
                        (count, spread, now_iso, name, speaker_id),
                    )
                else:
                    self._conn.execute(
                        "UPDATE speakers SET sample_count = ?, sample_spread = ?, "
                        "updated_at = ? WHERE id = ?",
                        (count, spread, now_iso, speaker_id),
                    )
                if self._storage.vec_loaded:
                    self._storage.upsert_vec(speaker_id, avg_emb)

            self._on_enroll()

    def delete_speaker(self, speaker_id: str):
        with self._lock:
            self._require_speaker(speaker_id)
            with transaction(self._conn):
                if self._storage.vec_loaded:
                    self._storage.delete_vec(speaker_id)
                self._conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))

    def rename_speaker(self, speaker_id: str, new_name: str):
        with self._lock, transaction(self._conn):
            self._require_speaker(speaker_id)
            self._conn.execute(
                "UPDATE speakers SET name = ?, updated_at = ? WHERE id = ?",
                (new_name, datetime.now().isoformat(), speaker_id),
            )

    def list_speakers(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, sample_count, sample_spread, created_at, updated_at "
                "FROM speakers"
            ).fetchall()
        return [self._row_to_speaker(row) for row in rows]

    def get_speaker(self, speaker_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, sample_count, sample_spread, created_at, updated_at "
                "FROM speakers WHERE id = ?",
                (speaker_id,),
            ).fetchone()
        return self._row_to_speaker(row) if row is not None else None

    def fetch_identify_candidate(self, query: np.ndarray) -> IdentifyCandidate | None:
        candidates = self.fetch_identify_candidates(query, limit=1)
        return candidates[0] if candidates else None

    def fetch_identify_candidates(
        self, query: np.ndarray, limit: int = 2
    ) -> list[IdentifyCandidate]:
        with self._lock:
            matches = self._find_best_matches(query, limit=limit)
            candidates: list[IdentifyCandidate] = []
            for speaker_id, similarity in matches:
                speaker_row = self._conn.execute(
                    "SELECT name, sample_count, sample_spread FROM speakers WHERE id = ?",
                    (speaker_id,),
                ).fetchone()
                if speaker_row is None:
                    continue

                emb_row = self._conn.execute(
                    "SELECT embedding FROM speaker_avg WHERE speaker_id = ?",
                    (speaker_id,),
                ).fetchone()
                enroll_emb = (
                    None if emb_row is None else blob_to_emb(emb_row["embedding"])
                )
                candidates.append(
                    IdentifyCandidate(
                        speaker_id=speaker_id,
                        name=speaker_row["name"],
                        sample_count=int(speaker_row["sample_count"]),
                        sample_spread=speaker_row["sample_spread"],
                        similarity=similarity,
                        enroll_emb=enroll_emb,
                    )
                )
            return candidates

    def _require_speaker(self, speaker_id: str):
        row = self._conn.execute(
            "SELECT id FROM speakers WHERE id = ?",
            (speaker_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Speaker {speaker_id} not found")

    def _find_best_match(self, query: np.ndarray) -> tuple[str | None, float]:
        matches = self._find_best_matches(query, limit=1)
        return matches[0] if matches else (None, 0.0)

    def _find_best_matches(self, query: np.ndarray, limit: int = 2) -> list[tuple[str, float]]:
        limit = max(1, int(limit))
        if self._storage.vec_loaded and self._storage.vec_table_dim is not None:
            try:
                rows = self._conn.execute(
                    "SELECT speaker_id, distance FROM speaker_vecs "
                    "WHERE avg_emb MATCH ? AND k = ?",
                    (self._serialize_for_vec(query), limit),
                ).fetchall()
                if rows:
                    return [
                        (row["speaker_id"], float(1.0 - row["distance"]))
                        for row in rows
                    ]
            except sqlite3.OperationalError:
                pass
        return self._python_cosine_scan(query, limit=limit)

    @staticmethod
    def _serialize_for_vec(query: np.ndarray) -> bytes:
        try:
            import sqlite_vec as _sv

            return _sv.serialize_float32(query.astype(np.float32).flatten().tolist())
        except (ImportError, AttributeError):
            import struct as _struct

            qflat = query.astype(np.float32).flatten()
            return _struct.pack(f"<{len(qflat)}f", *qflat)

    def _python_cosine_scan(
        self, query: np.ndarray, limit: int = 1
    ) -> list[tuple[str, float]]:
        rows = self._conn.execute(
            "SELECT speaker_id, embedding FROM speaker_avg"
        ).fetchall()
        if not rows:
            return []

        q = query.flatten().astype(np.float32)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0:
            return []

        ids = [row["speaker_id"] for row in rows]
        embs = np.stack([blob_to_emb(row["embedding"]) for row in rows])
        q_normed = q / q_norm
        emb_norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_normed = embs / np.where(emb_norms == 0, 1.0, emb_norms)
        similarities = embs_normed @ q_normed
        best_indices = np.argsort(similarities)[::-1][: max(1, int(limit))]
        return [(ids[int(idx)], float(similarities[int(idx)])) for idx in best_indices]

    def _recompute_avg_and_spread(
        self, speaker_id: str
    ) -> tuple[np.ndarray, float | None]:
        rows = self._conn.execute(
            "SELECT embedding FROM speaker_samples WHERE speaker_id = ?",
            (speaker_id,),
        ).fetchall()
        if not rows:
            raise ValueError(f"No samples for speaker {speaker_id}")

        arrays = [blob_to_emb(row["embedding"]) for row in rows]
        stacked = np.stack(arrays, axis=0)
        avg = stacked.mean(axis=0).astype(np.float32)
        if len(arrays) < 2:
            return avg, None

        avg_norm = np.linalg.norm(avg)
        if avg_norm == 0:
            return avg, None

        sims = []
        for sample in arrays:
            sample_norm = np.linalg.norm(sample)
            if sample_norm != 0:
                sims.append(float(np.dot(sample, avg) / (sample_norm * avg_norm)))
        if len(sims) < 2:
            return avg, None
        return avg, float(np.std(sims, ddof=0))

    @staticmethod
    def _row_to_speaker(row) -> dict:
        return {
            "id": row["id"],
            "name": row["name"],
            "sample_count": row["sample_count"],
            "sample_spread": row["sample_spread"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
