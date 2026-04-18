"""Persistent voiceprint database for speaker identification.

Storage backend: sqlite + sqlite-vec (vec0 virtual table with cosine metric).
Falls back to a Python-side cosine scan if sqlite-vec fails to load.

Schema
------
speakers(id TEXT PK, name TEXT, sample_count INT, created_at TEXT, updated_at TEXT)
speaker_samples(id INTEGER PK AUTOINCREMENT, speaker_id TEXT FK, embedding BLOB)
speaker_avg(speaker_id TEXT PK FK, embedding BLOB)
speaker_vecs  -- vec0 virtual table, created after first insert so dimension is known

WAL mode + busy_timeout give safe concurrent reads from multiple threads.
A process-level RLock serialises all writes (vec0 multi-statement updates are
not atomically isolated across threads without it).
"""

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_VEC_TABLE_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS speaker_vecs "
    "USING vec0(speaker_id TEXT PRIMARY KEY, avg_emb FLOAT[{dim}] distance_metric=cosine)"
)

_CORE_DDL = """
CREATE TABLE IF NOT EXISTS speakers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS speaker_samples (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id  TEXT NOT NULL REFERENCES speakers(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS speaker_avg (
    speaker_id  TEXT PRIMARY KEY REFERENCES speakers(id) ON DELETE CASCADE,
    embedding   BLOB NOT NULL
);
"""


def _emb_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).flatten().tobytes()


def _blob_to_emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class VoiceprintDB:
    """Thread-safe speaker database backed by sqlite + sqlite-vec.

    Public API is identical to the legacy .npy / index.json implementation so
    that main.py requires no changes.
    """

    def __init__(self, db_dir: str = "/data/voiceprints"):
        self.db_dir = Path(db_dir)
        self.db_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = str(self.db_dir / "voiceprints.db")
        self._lock = threading.RLock()
        self._vec_loaded = False
        self._vec_table_dim: Optional[int] = None

        self._conn = self._open_connection()
        self._init_schema()
        self._try_load_vec()
        self._maybe_migrate_legacy()

    # ------------------------------------------------------------------
    # Connection / schema helpers
    # ------------------------------------------------------------------

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path,
            check_same_thread=False,
            isolation_level=None,  # autocommit off; we manage transactions manually
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        with self._lock:
            self._conn.executescript(_CORE_DDL)

    def _try_load_vec(self):
        """Attempt to load the sqlite-vec extension.  On failure log a warning
        and continue — identify() will use a Python-side cosine fallback."""
        try:
            import sqlite_vec  # noqa: PLC0415

            self._conn.enable_load_extension(True)
            sqlite_vec.load(self._conn)
            self._conn.enable_load_extension(False)
            self._vec_loaded = True
            logger.debug("sqlite-vec %s loaded", sqlite_vec.__version__)

            # If the virtual table already exists, read back its dimension from
            # a stored average embedding so we can re-create it if needed.
            row = self._conn.execute(
                "SELECT embedding FROM speaker_avg LIMIT 1"
            ).fetchone()
            if row:
                dim = len(_blob_to_emb(row["embedding"]))
                self._ensure_vec_table(dim)
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "sqlite-vec not available — falling back to Python cosine scan: %s", exc
            )
            self._vec_loaded = False

    def _ensure_vec_table(self, dim: int):
        """Create the vec0 virtual table for *dim*-dimensional embeddings if it
        does not yet exist.  Must only be called when sqlite-vec is loaded."""
        if self._vec_table_dim == dim:
            return
        try:
            self._conn.execute(_VEC_TABLE_DDL.format(dim=dim))
            self._vec_table_dim = dim
        except sqlite3.OperationalError:
            # Table already exists with the same dim — that is fine.
            self._vec_table_dim = dim

    # ------------------------------------------------------------------
    # Legacy .npy migration
    # ------------------------------------------------------------------

    def _maybe_migrate_legacy(self):
        index_file = self.db_dir / "index.json"
        if not index_file.exists():
            return

        count = self._conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]
        if count != 0:
            return

        logger.info("Migrating legacy .npy voiceprint store …")
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read index.json, skipping migration: %s", exc)
            return

        speakers = index.get("speakers", {})
        migrated = 0

        with self._lock:
            self._conn.execute("BEGIN")
            try:
                for spk_id, info in speakers.items():
                    avg_path = self.db_dir / f"{spk_id}_avg.npy"
                    samples_path = self.db_dir / f"{spk_id}_samples.npy"

                    if not avg_path.exists():
                        logger.warning(
                            "Missing %s, skipping speaker %s", avg_path, spk_id
                        )
                        continue

                    avg_emb = (
                        np.load(str(avg_path), allow_pickle=False)
                        .flatten()
                        .astype(np.float32)
                    )

                    self._conn.execute(
                        "INSERT INTO speakers(id, name, sample_count, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (
                            spk_id,
                            info.get("name", spk_id),
                            info.get("sample_count", 1),
                            info.get("created_at", datetime.now().isoformat()),
                            info.get("updated_at", datetime.now().isoformat()),
                        ),
                    )
                    self._conn.execute(
                        "INSERT INTO speaker_avg(speaker_id, embedding) VALUES (?, ?)",
                        (spk_id, _emb_to_blob(avg_emb)),
                    )

                    if samples_path.exists():
                        samples = np.load(str(samples_path), allow_pickle=False)
                        if samples.ndim == 1:
                            samples = samples.reshape(1, -1)
                        for row in samples:
                            self._conn.execute(
                                "INSERT INTO speaker_samples(speaker_id, embedding) VALUES (?, ?)",
                                (spk_id, _emb_to_blob(row.astype(np.float32))),
                            )

                    if self._vec_loaded:
                        self._ensure_vec_table(len(avg_emb))
                        self._upsert_vec(spk_id, avg_emb)

                    migrated += 1

                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

        index_file.rename(index_file.with_suffix(".json.migrated.bak"))
        logger.info("Migrated %d speakers from legacy .npy store", migrated)

    # ------------------------------------------------------------------
    # Internal helpers (caller must hold _lock)
    # ------------------------------------------------------------------

    def _upsert_vec(self, speaker_id: str, avg_emb: np.ndarray):
        """Insert or update a speaker's average embedding in the vec0 table.

        vec0 does not support INSERT OR REPLACE on its primary key, so we use
        UPDATE when the row already exists and INSERT when it does not.
        """
        existing = self._conn.execute(
            "SELECT 1 FROM speaker_vecs WHERE speaker_id = ?", (speaker_id,)
        ).fetchone()
        if existing:
            self._conn.execute(
                "UPDATE speaker_vecs SET avg_emb = ? WHERE speaker_id = ?",
                (_emb_to_blob(avg_emb), speaker_id),
            )
        else:
            self._conn.execute(
                "INSERT INTO speaker_vecs(speaker_id, avg_emb) VALUES (?, ?)",
                (speaker_id, _emb_to_blob(avg_emb)),
            )

    def _delete_vec(self, speaker_id: str):
        self._conn.execute(
            "DELETE FROM speaker_vecs WHERE speaker_id = ?", (speaker_id,)
        )

    def _recompute_avg(self, speaker_id: str) -> np.ndarray:
        """Recompute mean embedding from all samples for *speaker_id*."""
        rows = self._conn.execute(
            "SELECT embedding FROM speaker_samples WHERE speaker_id = ?",
            (speaker_id,),
        ).fetchall()
        if not rows:
            raise ValueError(f"No samples for speaker {speaker_id}")
        arrays = [_blob_to_emb(r["embedding"]) for r in rows]
        return np.stack(arrays, axis=0).mean(axis=0).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API — mutations hold _lock and run in explicit transactions
    # ------------------------------------------------------------------

    def add_speaker(self, name: str, embedding: np.ndarray) -> str:
        """Register a new speaker with a name and initial embedding.

        Returns the generated ``spk_xxx`` id.
        """
        speaker_id = f"spk_{uuid.uuid4().hex[:8]}"
        emb = embedding.flatten().astype(np.float32)
        now = datetime.now().isoformat()

        with self._lock:
            if self._vec_loaded:
                self._ensure_vec_table(len(emb))

            self._conn.execute("BEGIN")
            try:
                self._conn.execute(
                    "INSERT INTO speakers(id, name, sample_count, created_at, updated_at) "
                    "VALUES (?, ?, 1, ?, ?)",
                    (speaker_id, name, now, now),
                )
                self._conn.execute(
                    "INSERT INTO speaker_samples(speaker_id, embedding) VALUES (?, ?)",
                    (speaker_id, _emb_to_blob(emb)),
                )
                self._conn.execute(
                    "INSERT INTO speaker_avg(speaker_id, embedding) VALUES (?, ?)",
                    (speaker_id, _emb_to_blob(emb)),
                )
                if self._vec_loaded:
                    self._upsert_vec(speaker_id, emb)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

        return speaker_id

    def update_speaker(
        self, speaker_id: str, new_embedding: np.ndarray, name: Optional[str] = None
    ):
        """Append a new sample and recompute the mean embedding."""
        emb = new_embedding.flatten().astype(np.float32)

        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Speaker {speaker_id} not found")

            self._conn.execute("BEGIN")
            try:
                self._conn.execute(
                    "INSERT INTO speaker_samples(speaker_id, embedding) VALUES (?, ?)",
                    (speaker_id, _emb_to_blob(emb)),
                )
                avg_emb = self._recompute_avg(speaker_id)
                self._conn.execute(
                    "INSERT OR REPLACE INTO speaker_avg(speaker_id, embedding) VALUES (?, ?)",
                    (speaker_id, _emb_to_blob(avg_emb)),
                )
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM speaker_samples WHERE speaker_id = ?",
                    (speaker_id,),
                ).fetchone()[0]
                update_fields = "sample_count = ?, updated_at = ?"
                params: list = [count, datetime.now().isoformat()]
                if name is not None:
                    update_fields += ", name = ?"
                    params.append(name)
                params.append(speaker_id)
                self._conn.execute(
                    f"UPDATE speakers SET {update_fields} WHERE id = ?",
                    params,
                )
                if self._vec_loaded:
                    self._upsert_vec(speaker_id, avg_emb)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def delete_speaker(self, speaker_id: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Speaker {speaker_id} not found")

            self._conn.execute("BEGIN")
            try:
                if self._vec_loaded:
                    self._delete_vec(speaker_id)
                # CASCADE handles speaker_samples and speaker_avg
                self._conn.execute("DELETE FROM speakers WHERE id = ?", (speaker_id,))
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def rename_speaker(self, speaker_id: str, new_name: str):
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Speaker {speaker_id} not found")

            self._conn.execute("BEGIN")
            try:
                self._conn.execute(
                    "UPDATE speakers SET name = ?, updated_at = ? WHERE id = ?",
                    (new_name, datetime.now().isoformat(), speaker_id),
                )
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------

    def identify(
        self, embedding: np.ndarray, threshold: float = 0.75
    ) -> tuple[Optional[str], Optional[str], float]:
        """Return ``(speaker_id, speaker_name, similarity)`` for the closest match.

        Returns ``(None, None, best_similarity)`` when no speaker exceeds
        *threshold* or the database is empty.  ``best_similarity`` is always
        the raw best score found (existing behaviour), even when below threshold.
        """
        query = embedding.flatten().astype(np.float32)

        with self._lock:
            # Fast path: sqlite-vec cosine ANN
            if self._vec_loaded and self._vec_table_dim is not None:
                try:
                    row = self._conn.execute(
                        "SELECT speaker_id, distance FROM speaker_vecs "
                        "WHERE avg_emb MATCH ? AND k = 1",
                        (query.tobytes(),),
                    ).fetchone()
                    if row is None:
                        return None, None, 0.0
                    best_id = row["speaker_id"]
                    # distance = 1 - cosine_similarity for cosine metric
                    best_sim = float(1.0 - row["distance"])
                except sqlite3.OperationalError:
                    # vec table might be empty or not yet created
                    best_id, best_sim = self._python_cosine_scan(query)
            else:
                best_id, best_sim = self._python_cosine_scan(query)

        if best_id is None:
            return None, None, 0.0

        if best_sim >= threshold:
            spk_row = self._conn.execute(
                "SELECT name FROM speakers WHERE id = ?", (best_id,)
            ).fetchone()
            if spk_row is not None:
                return best_id, spk_row["name"], best_sim

        return None, None, best_sim

    def _python_cosine_scan(self, query: np.ndarray) -> tuple[Optional[str], float]:
        """Full-scan cosine similarity over speaker_avg (fallback path)."""
        rows = self._conn.execute(
            "SELECT speaker_id, embedding FROM speaker_avg"
        ).fetchall()
        if not rows:
            return None, 0.0

        best_id: Optional[str] = None
        best_sim = -1.0
        q_norm = np.linalg.norm(query)
        if q_norm == 0:
            return None, 0.0

        for row in rows:
            avg = _blob_to_emb(row["embedding"])
            a_norm = np.linalg.norm(avg)
            if a_norm == 0:
                continue
            sim = float(np.dot(query, avg) / (q_norm * a_norm))
            if sim > best_sim:
                best_sim = sim
                best_id = row["speaker_id"]

        return best_id, best_sim

    def list_speakers(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, sample_count, created_at, updated_at FROM speakers"
            ).fetchall()
        return [
            {
                "id": r["id"],
                "name": r["name"],
                "sample_count": r["sample_count"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    def get_speaker(self, speaker_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, sample_count, created_at, updated_at "
                "FROM speakers WHERE id = ?",
                (speaker_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "sample_count": row["sample_count"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
