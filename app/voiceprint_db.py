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

import base64
import glob as _glob
import json
import logging
import os
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# 嵌入向量维度（WeSpeaker ResNet34 默认 256）。通过 env 覆盖以支持模型切换。
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "256"))

_VEC_TABLE_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS speaker_vecs "
    "USING vec0(speaker_id TEXT PRIMARY KEY, avg_emb FLOAT[{dim}] distance_metric=cosine)"
)

_CORE_DDL = """
CREATE TABLE IF NOT EXISTS speakers (
    id           TEXT PRIMARY KEY,
    name         TEXT NOT NULL,
    sample_count INTEGER NOT NULL DEFAULT 0,
    sample_spread REAL,   -- std of cos(sample_i, avg); NULL when sample_count <= 1
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

# Threshold tuning knobs (adaptive identification).
# A freshly enrolled speaker has just one sample, so its averaged embedding is
# the single noisy sample — we loosen the match threshold by this much to
# accept the inevitable cross-session drift.
_SINGLE_SAMPLE_RELAXATION = 0.05  # 0.75 - 0.05 = 0.70 by default
# For multi-sample speakers we compute the std of cos(sample_i, avg). The
# dynamic threshold relaxes by k * std, capped at _SPREAD_RELAXATION_CAP so a
# pathologically noisy cluster can't pull the threshold arbitrarily low.
_SPREAD_RELAXATION_K = 3.0
_SPREAD_RELAXATION_CAP = 0.10
# Absolute floor — never accept a match below this, regardless of per-speaker
# relaxation. Guards against false positives from degenerate clusters.
_ABSOLUTE_FLOOR = 0.60


def _emb_to_blob(arr: np.ndarray) -> bytes:
    return arr.astype(np.float32).flatten().tobytes()


def _blob_to_emb(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


class ASNormScorer:
    """AS-norm score normalization using a cohort of impostor embeddings."""

    def __init__(self, cohort: np.ndarray, top_n: int = 200):
        norms = np.linalg.norm(cohort, axis=1, keepdims=True)
        self._cohort = cohort / (norms + 1e-8)  # (N, 256), L2-normed
        self._top_n = min(top_n, len(cohort))

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
        if len(self._cohort) < 10:
            return raw  # not enough cohort → fall back to raw cosine
        mean_e, std_e = self._cohort_stats(enroll_emb)
        mean_t, std_t = self._cohort_stats(test_emb)
        return 0.5 * ((raw - mean_e) / std_e + (raw - mean_t) / std_t)


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
        self._vec_table_dim: int | None = None

        self._asnorm: ASNormScorer | None = None
        self._asnorm_threshold: float = 0.5  # AS-norm operating point

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
            # Lazy migration: older voscript installs had no sample_spread
            # column. ADD COLUMN keeps existing rows; the value defaults to
            # NULL which the identify() logic treats as "unknown → use base
            # threshold" (still safer than guessing).
            cols = {
                row["name"] for row in self._conn.execute("PRAGMA table_info(speakers)")
            }
            if "sample_spread" not in cols:
                logger.info("voiceprint_db: adding speakers.sample_spread column")
                self._conn.execute("ALTER TABLE speakers ADD COLUMN sample_spread REAL")

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

    def _recompute_avg_and_spread(
        self, speaker_id: str
    ) -> tuple[np.ndarray, float | None]:
        """Recompute mean embedding + intra-cluster cosine spread.

        Returns ``(avg_embedding, spread)`` where ``spread`` is the standard
        deviation of the cosine similarities between each individual sample
        and the recomputed mean. When the cluster has 1 or 0 samples the
        spread is ``None`` (undefined — the caller stores NULL).
        """
        rows = self._conn.execute(
            "SELECT embedding FROM speaker_samples WHERE speaker_id = ?",
            (speaker_id,),
        ).fetchall()
        if not rows:
            raise ValueError(f"No samples for speaker {speaker_id}")
        arrays = [_blob_to_emb(r["embedding"]) for r in rows]
        stacked = np.stack(arrays, axis=0)
        avg = stacked.mean(axis=0).astype(np.float32)
        if len(arrays) < 2:
            return avg, None
        # Per-sample cosine to the mean, then std of those cosines.
        a_norm = np.linalg.norm(avg)
        if a_norm == 0:
            return avg, None
        sims = []
        for s in arrays:
            s_norm = np.linalg.norm(s)
            if s_norm == 0:
                continue
            sims.append(float(np.dot(s, avg) / (s_norm * a_norm)))
        if len(sims) < 2:
            return avg, None
        return avg, float(np.std(sims, ddof=0))

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
                    "INSERT INTO speakers(id, name, sample_count, sample_spread, "
                    "created_at, updated_at) VALUES (?, ?, 1, NULL, ?, ?)",
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
        self, speaker_id: str, new_embedding: np.ndarray, name: str | None = None
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
                avg_emb, spread = self._recompute_avg_and_spread(speaker_id)
                self._conn.execute(
                    "INSERT OR REPLACE INTO speaker_avg(speaker_id, embedding) VALUES (?, ?)",
                    (speaker_id, _emb_to_blob(avg_emb)),
                )
                count = self._conn.execute(
                    "SELECT COUNT(*) FROM speaker_samples WHERE speaker_id = ?",
                    (speaker_id,),
                ).fetchone()[0]
                now_iso = datetime.now().isoformat()
                # 拆成两条静态 SQL，避免动态 f-string 拼接在后续维护中引入注入风险。
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

        Returns ``(None, None, best_similarity)`` when the best candidate is
        below its effective threshold. ``best_similarity`` is always the raw
        best score (existing behaviour), even when rejected.
        """
        query = embedding.flatten().astype(np.float32)

        # 零向量防御：AS-norm 分支对全 0 embedding 归一化分=0，与 raw cosine 语义冲突。
        if float(np.linalg.norm(query)) < 1e-6:
            return None, None, 0.0

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

            spk_row = self._conn.execute(
                "SELECT name, sample_count, sample_spread FROM speakers WHERE id = ?",
                (best_id,),
            ).fetchone()

            # AS-norm scoring: replace raw cosine with normalized score when
            # cohort available. When active we bypass the adaptive relaxation
            # logic — AS-norm already handles speaker variability — and use
            # the fixed AS-norm operating point.
            asnorm_active = False
            if self._asnorm is not None:
                best_emb_row = self._conn.execute(
                    "SELECT embedding FROM speaker_avg WHERE speaker_id = ?",
                    (best_id,),
                ).fetchone()
                if best_emb_row is not None:
                    enroll_emb = _blob_to_emb(best_emb_row["embedding"])
                    best_sim = self._asnorm.score(enroll_emb, query)
                    asnorm_active = True

        if spk_row is None:
            # Race: vec table still references a row the caller deleted.
            return None, None, best_sim

        if asnorm_active:
            effective = self._asnorm_threshold
            logger.debug(
                "identify[asnorm]: best=%s normalized_sim=%.4f threshold=%.3f",
                best_id,
                best_sim,
                effective,
            )
        else:
            effective = self._effective_threshold(
                base=threshold,
                sample_count=int(spk_row["sample_count"]),
                sample_spread=spk_row["sample_spread"],
            )
            logger.debug(
                "identify: best=%s best_sim=%.4f base=%.3f effective=%.3f "
                "(n=%d, spread=%s)",
                best_id,
                best_sim,
                threshold,
                effective,
                spk_row["sample_count"],
                spk_row["sample_spread"],
            )

        if best_sim >= effective:
            return best_id, spk_row["name"], best_sim
        return None, None, best_sim

    @staticmethod
    def _effective_threshold(
        base: float, sample_count: int, sample_spread: float | None
    ) -> float:
        """Adaptive threshold per-candidate.

        - 1 sample → base - _SINGLE_SAMPLE_RELAXATION (fixed relax)
        - ≥2 samples, NULL spread (legacy row) → base
        - ≥2 samples, known spread → base - min(k*spread, _SPREAD_RELAXATION_CAP)

        Result is clamped to ``[_ABSOLUTE_FLOOR, base]``.
        """
        if sample_count <= 1 or sample_spread is None:
            if sample_count <= 1:
                dyn = base - _SINGLE_SAMPLE_RELAXATION
            else:
                dyn = base
        else:
            relax = min(
                _SPREAD_RELAXATION_K * float(sample_spread), _SPREAD_RELAXATION_CAP
            )
            dyn = base - relax
        return max(_ABSOLUTE_FLOOR, min(base, dyn))

    def _python_cosine_scan(self, query: np.ndarray) -> tuple[str | None, float]:
        """Full-scan cosine similarity over speaker_avg (fallback path)."""
        rows = self._conn.execute(
            "SELECT speaker_id, embedding FROM speaker_avg"
        ).fetchall()
        if not rows:
            return None, 0.0

        best_id: str | None = None
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
                "SELECT id, name, sample_count, sample_spread, created_at, updated_at "
                "FROM speakers"
            ).fetchall()
        return [self._row_to_speaker(r) for r in rows]

    def get_speaker(self, speaker_id: str) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, sample_count, sample_spread, created_at, updated_at "
                "FROM speakers WHERE id = ?",
                (speaker_id,),
            ).fetchone()
        return self._row_to_speaker(row) if row is not None else None

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

    # ------------------------------------------------------------------
    # AS-norm cohort management
    # ------------------------------------------------------------------

    def load_cohort(self, cohort_path: str, top_n: int = 200):
        """Load a pre-saved cohort numpy array (.npy, shape [N, 256])."""
        arr = np.load(cohort_path, allow_pickle=False).astype(np.float32)
        if arr.ndim != 2:
            raise ValueError(f"Cohort must be 2D, got {arr.ndim}")
        self._asnorm = ASNormScorer(arr, top_n=top_n)
        logger.info("AS-norm cohort loaded: %d speakers, top_n=%d", len(arr), top_n)

    def build_cohort_from_transcriptions(
        self, transcriptions_dir: str, save_path: str | None = None
    ) -> int:
        """Build a cohort from speaker_embeddings in existing result.json files.

        Returns the number of cohort embeddings collected. The number of
        skipped / corrupted files can be retrieved from
        ``self.last_cohort_skipped`` after the call (see [CQ-M10]).

        Two persistence formats are supported:
        - ``result.json["speaker_embeddings"]`` dict keyed by speaker label,
          values either a Python list (``[float, ...]``) or a base64-encoded
          float32 byte string.
        - ``emb_<SPEAKER_LABEL>.npy`` files sibling to ``result.json`` — this
          is how the current pipeline persists embeddings on disk. Used as a
          fallback when ``speaker_embeddings`` isn't present in the JSON.
        """
        embs = []
        skipped_files = 0
        expected_shape = (EMBEDDING_DIM,)
        for f in _glob.glob(str(Path(transcriptions_dir) / "*/result.json")):
            try:
                with open(f) as fh:
                    d = json.load(fh)
                se = d.get("speaker_embeddings", {})
                added_from_json = 0
                for v in se.values():
                    if isinstance(v, list):
                        arr = np.array(v, dtype=np.float32)
                    elif isinstance(v, str):
                        arr = np.frombuffer(base64.b64decode(v), dtype=np.float32)
                    else:
                        continue
                    if arr.shape == expected_shape:
                        embs.append(arr)
                        added_from_json += 1

                # Fallback: sibling emb_*.npy files (pipeline's on-disk format)
                if added_from_json == 0:
                    tr_dir = Path(f).parent
                    for npy_path in tr_dir.glob("emb_*.npy"):
                        try:
                            arr = (
                                np.load(str(npy_path), allow_pickle=False)
                                .flatten()
                                .astype(np.float32)
                            )
                            if arr.shape == expected_shape:
                                embs.append(arr)
                        except Exception as exc:
                            skipped_files += 1
                            logger.warning(
                                "build_cohort: skip %s due to load error: %s",
                                npy_path,
                                exc,
                            )
                            continue
            except Exception as exc:
                skipped_files += 1
                logger.warning("build_cohort: skip %s: %s", f, exc)
                continue
        self.last_cohort_skipped = skipped_files

        if not embs:
            logger.warning(
                "build_cohort_from_transcriptions: no embeddings found in %s",
                transcriptions_dir,
            )
            return 0

        cohort = np.stack(embs, axis=0)
        if save_path:
            np.save(save_path, cohort)
            logger.info("Cohort saved: %d embeddings → %s", len(cohort), save_path)
        self._asnorm = ASNormScorer(cohort, top_n=min(200, len(cohort)))
        logger.info(
            "AS-norm cohort built from transcriptions: %d embeddings",
            len(cohort),
        )
        return len(cohort)

    def set_asnorm_threshold(self, threshold: float):
        self._asnorm_threshold = threshold
