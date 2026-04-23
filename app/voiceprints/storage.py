"""SQLite/bootstrap helpers for the voiceprint store."""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_VEC_TABLE_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS speaker_vecs "
    "USING vec0(speaker_id TEXT PRIMARY KEY, avg_emb FLOAT[{dim}] distance_metric=cosine)"
)

_CORE_DDL = """
CREATE TABLE IF NOT EXISTS speakers (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    sample_spread REAL,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL
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


def emb_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).flatten().tobytes()


def blob_to_emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


@contextmanager
def transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN")
    try:
        yield
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


class VoiceprintStorage:
    """Owns sqlite connection lifecycle, schema bootstrap and legacy migration."""

    def __init__(self, db_dir: str | Path, lock):
        self.db_dir = Path(db_dir)
        self._lock = lock
        self.db_path = str(self.db_dir / "voiceprints.db")
        self.conn = self._open_connection()
        self.vec_loaded = False
        self.vec_table_dim: int | None = None

    def bootstrap(self):
        self._init_schema()
        self._try_load_vec()
        self._maybe_migrate_legacy()

    def _open_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        with self._lock:
            self.conn.executescript(_CORE_DDL)
            cols = {row["name"] for row in self.conn.execute("PRAGMA table_info(speakers)")}
            if "sample_spread" not in cols:
                logger.info("voiceprint_db: adding speakers.sample_spread column")
                self.conn.execute("ALTER TABLE speakers ADD COLUMN sample_spread REAL")

    def _try_load_vec(self):
        try:
            import sqlite_vec  # noqa: PLC0415

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
            self.vec_loaded = True
            logger.debug("sqlite-vec %s loaded", sqlite_vec.__version__)

            row = self.conn.execute("SELECT embedding FROM speaker_avg LIMIT 1").fetchone()
            if row is not None:
                self.ensure_vec_table(len(blob_to_emb(row["embedding"])))
        except Exception as exc:  # pragma: no cover
            logger.warning(
                "sqlite-vec not available — falling back to Python cosine scan: %s",
                exc,
            )
            self.vec_loaded = False

    def ensure_vec_table(self, dim: int):
        if self.vec_table_dim == dim:
            return
        try:
            self.conn.execute(_VEC_TABLE_DDL.format(dim=dim))
        except sqlite3.OperationalError:
            pass
        self.vec_table_dim = dim

    def upsert_vec(self, speaker_id: str, avg_emb: np.ndarray):
        existing = self.conn.execute(
            "SELECT 1 FROM speaker_vecs WHERE speaker_id = ?",
            (speaker_id,),
        ).fetchone()
        if existing:
            self.conn.execute(
                "UPDATE speaker_vecs SET avg_emb = ? WHERE speaker_id = ?",
                (emb_to_blob(avg_emb), speaker_id),
            )
            return
        self.conn.execute(
            "INSERT INTO speaker_vecs(speaker_id, avg_emb) VALUES (?, ?)",
            (speaker_id, emb_to_blob(avg_emb)),
        )

    def delete_vec(self, speaker_id: str):
        self.conn.execute("DELETE FROM speaker_vecs WHERE speaker_id = ?", (speaker_id,))

    def _maybe_migrate_legacy(self):
        index_file = self.db_dir / "index.json"
        if not index_file.exists():
            return
        count = self.conn.execute("SELECT COUNT(*) FROM speakers").fetchone()[0]
        if count != 0:
            return

        logger.info("Migrating legacy .npy voiceprint store …")
        try:
            index = json.loads(index_file.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read index.json, skipping migration: %s", exc)
            return

        migrated = 0
        speakers = index.get("speakers", {})
        with self._lock, transaction(self.conn):
            for speaker_id, info in speakers.items():
                avg_path = self.db_dir / f"{speaker_id}_avg.npy"
                samples_path = self.db_dir / f"{speaker_id}_samples.npy"
                if not avg_path.exists():
                    logger.warning("Missing %s, skipping speaker %s", avg_path, speaker_id)
                    continue

                avg_emb = np.load(str(avg_path), allow_pickle=False).flatten().astype(np.float32)
                now = datetime.now().isoformat()
                self.conn.execute(
                    "INSERT INTO speakers(id, name, sample_count, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        speaker_id,
                        info.get("name", speaker_id),
                        info.get("sample_count", 1),
                        info.get("created_at", now),
                        info.get("updated_at", now),
                    ),
                )
                self.conn.execute(
                    "INSERT INTO speaker_avg(speaker_id, embedding) VALUES (?, ?)",
                    (speaker_id, emb_to_blob(avg_emb)),
                )
                if samples_path.exists():
                    samples = np.load(str(samples_path), allow_pickle=False)
                    if samples.ndim == 1:
                        samples = samples.reshape(1, -1)
                    for row in samples:
                        self.conn.execute(
                            "INSERT INTO speaker_samples(speaker_id, embedding) VALUES (?, ?)",
                            (speaker_id, emb_to_blob(row.astype(np.float32))),
                        )
                if self.vec_loaded:
                    self.ensure_vec_table(len(avg_emb))
                    self.upsert_vec(speaker_id, avg_emb)
                migrated += 1

        index_file.rename(index_file.with_suffix(".json.migrated.bak"))
        logger.info("Migrated %d speakers from legacy .npy store", migrated)
