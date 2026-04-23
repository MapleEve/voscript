"""TEST-C2 / TEST-C4: VoiceprintDB core logic & AS-norm regressions.

Covers:
- add_speaker → identify round-trip with a real on-disk SQLite database
- zero-vector defence in identify()
- adaptive threshold: single-sample speaker accepts at ~0.70 (base 0.75)
- deduplication by name (same-name re-enrol updates instead of inserting)
- AS-norm only kicks in when cohort >= 10 embeddings
- update_speaker uses static SQL with/without the optional name argument

The tests rely on sqlite-vec when available; every check also passes on the
Python cosine fallback so they don't over-specify the backend.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_db(db_dir: Path):
    """Import voiceprints.db freshly and return a brand-new VoiceprintDB.

    A fresh import side-steps the stub registered in conftest.py — we want the
    production class backed by sqlite + (optionally) sqlite-vec.
    """
    for name in list(sys.modules):
        if name == "voiceprints" or name.startswith("voiceprints."):
            sys.modules.pop(name, None)
    mod = importlib.import_module("voiceprints.db")
    db_dir.mkdir(parents=True, exist_ok=True)
    return mod.VoiceprintDB(str(db_dir)), mod


def _unit_vec(seed: int, dim: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-9
    return v


# ---------------------------------------------------------------------------
# TEST-C2 – core add / identify behaviour
# ---------------------------------------------------------------------------


def test_add_speaker_and_identify(tmp_path):
    """add_speaker then identify with the same embedding returns that speaker."""
    db, _mod = _fresh_db(tmp_path / "vp")
    emb = _unit_vec(1)
    sid = db.add_speaker("alice", emb)
    assert sid.startswith("spk_")

    got_id, got_name, sim = db.identify(emb)
    assert got_id == sid
    assert got_name == "alice"
    assert sim >= 0.99  # cosine with itself ≈ 1.0 (minus float rounding)


def test_identify_zero_vector_returns_none(tmp_path):
    """All-zero embedding is a sentinel that must short-circuit to (None, None, 0.0)."""
    db, _mod = _fresh_db(tmp_path / "vp")
    db.add_speaker("bob", _unit_vec(2))
    zero = np.zeros(256, dtype=np.float32)
    assert db.identify(zero) == (None, None, 0.0)


def test_adaptive_threshold_single_sample(tmp_path):
    """Single-sample speakers must accept matches at sim ≈ 0.70 (base 0.75 − 0.05)."""
    db, mod = _fresh_db(tmp_path / "vp")

    # Compute the effective threshold the class itself uses so we don't
    # duplicate the constant.
    effective = mod.VoiceprintDB._effective_threshold(
        base=0.75, sample_count=1, sample_spread=None
    )
    assert (
        0.695 <= effective <= 0.705
    ), f"single-sample relaxation should yield ~0.70, got {effective}"

    # And the live identify() honours it: a slightly perturbed embedding
    # at similarity ≈ 0.72 must be accepted for a one-sample speaker, which
    # would have been rejected under the strict 0.75 threshold.
    enroll = _unit_vec(3)
    sid = db.add_speaker("carol", enroll)

    # Construct a query with cosine similarity ≈ 0.72 by mixing enroll with
    # an orthogonal vector.
    orth = _unit_vec(4)
    orth -= (orth @ enroll) * enroll  # Gram-Schmidt → orthogonal to enroll
    orth /= np.linalg.norm(orth) + 1e-9
    alpha = 0.72
    query = alpha * enroll + np.sqrt(max(0.0, 1 - alpha**2)) * orth
    query = query.astype(np.float32)
    query /= np.linalg.norm(query) + 1e-9

    got_id, got_name, sim = db.identify(query)
    assert got_id == sid, (
        f"expected match (sim={sim:.3f}, effective={effective:.3f}) — "
        "single-sample relaxation missing"
    )
    assert 0.70 <= sim <= 0.75


def test_add_speaker_dedup_by_name(tmp_path):
    """Adding a speaker with an existing (case-insensitive) name updates, not duplicates."""
    db, _mod = _fresh_db(tmp_path / "vp")
    first_id = db.add_speaker("Dana", _unit_vec(5))
    second_id = db.add_speaker("dana", _unit_vec(6))  # different case on purpose
    assert first_id == second_id, "dedup must reuse the existing speaker id"

    speakers = db.list_speakers()
    assert len(speakers) == 1
    # sample_count should reflect two samples after the dedup path took the
    # update_speaker branch.
    assert speakers[0]["sample_count"] == 2


def test_asnorm_active_only_when_cohort_ge_10(tmp_path):
    """Cohort size < 10 must NOT activate AS-norm (would invert thresholds)."""
    db, mod = _fresh_db(tmp_path / "vp")
    enroll = _unit_vec(7)
    db.add_speaker("erin", enroll)

    # Build a tiny cohort (size 5). AS-norm will be installed but identify()
    # must treat its output as a raw cosine and keep using the adaptive
    # threshold (not the 0.5 AS-norm operating point).
    small_cohort = np.stack([_unit_vec(100 + i) for i in range(5)])
    db._asnorm = mod.ASNormScorer(small_cohort, top_n=5)

    # Query identical to enroll → raw cosine ≈ 1.0 → must match.
    got_id, got_name, sim = db.identify(enroll)
    assert got_id is not None
    assert got_name == "erin"
    assert (
        sim >= 0.99
    ), f"cohort<10 should leave raw cosine untouched (got sim={sim:.3f})"

    # Sanity: construct a low-similarity query that would be accepted by
    # the AS-norm operating threshold (0.5) but rejected by the adaptive
    # cosine threshold. It must be rejected — proving the guard works.
    # Use sim ≈ 0.55 (below single-sample relaxation floor of 0.70).
    orth = _unit_vec(8)
    orth -= (orth @ enroll) * enroll
    orth /= np.linalg.norm(orth) + 1e-9
    low = 0.55 * enroll + np.sqrt(1 - 0.55**2) * orth
    low = low.astype(np.float32)
    low /= np.linalg.norm(low) + 1e-9

    got_id, _, sim = db.identify(low)
    assert (
        got_id is None
    ), f"cohort<10 must still reject sub-threshold raw cosine (sim={sim:.3f})"


def test_update_speaker_static_sql(tmp_path):
    """update_speaker works both with and without the optional name kwarg."""
    db, _mod = _fresh_db(tmp_path / "vp")
    sid = db.add_speaker("frank", _unit_vec(9))

    # Update without renaming.
    db.update_speaker(sid, _unit_vec(10))
    row = db.get_speaker(sid)
    assert row["name"] == "frank"
    assert row["sample_count"] == 2

    # Update with rename.
    db.update_speaker(sid, _unit_vec(11), name="Franklin")
    row = db.get_speaker(sid)
    assert row["name"] == "Franklin"
    assert row["sample_count"] == 3

    # A second no-rename update must leave the new name intact (regression
    # guard: the static-SQL refactor must not clobber the name column).
    db.update_speaker(sid, _unit_vec(12))
    row = db.get_speaker(sid)
    assert row["name"] == "Franklin"
    assert row["sample_count"] == 4


# ---------------------------------------------------------------------------
# TEST-C4 – AS-norm active flag regression
#
# TEST-C4 was folded into TEST-C2 per the task brief; the dedicated checks
# are ``test_asnorm_active_only_when_cohort_ge_10`` above and the explicit
# effective-threshold assertion below, which pins the published contract of
# _effective_threshold so future refactors don't drift the knobs.
# ---------------------------------------------------------------------------


def test_effective_threshold_pure_function(tmp_path):
    """Lock down the pure function that implements the adaptive threshold."""
    _db, mod = _fresh_db(tmp_path / "vp")
    f = mod.VoiceprintDB._effective_threshold

    # Single sample → base − SINGLE_SAMPLE_RELAXATION (0.05) → 0.70
    assert abs(f(0.75, 1, None) - 0.70) < 1e-9

    # Two+ samples with NULL spread (legacy) → base
    assert f(0.75, 3, None) == 0.75

    # Two+ samples with tiny spread → base − k * spread, clamped at CAP
    assert f(0.75, 5, 0.01) == pytest.approx(0.75 - 3.0 * 0.01, abs=1e-9)

    # Two+ samples with huge spread → relaxation clamps at 0.10 → 0.65
    assert f(0.75, 5, 1.0) == pytest.approx(0.65, abs=1e-9)

    # Absolute floor: relaxation can never drop below 0.60
    assert f(0.65, 5, 1.0) == pytest.approx(0.60, abs=1e-9)
