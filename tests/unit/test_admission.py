"""Tests for application-level transcription admission policy."""

from __future__ import annotations

import pytest

from application.admission import (
    AdmissionBudget,
    AdmissionRejectedError,
    RuntimeAdmissionSnapshot,
    admit_transcription_in_flight,
    ensure_transcription_admitted,
    find_in_flight_transcription,
    release_transcription_admission,
    reserve_transcription_admission,
)
import infra.job_runtime as job_runtime


def test_transcription_admission_allows_budget_headroom():
    ensure_transcription_admitted(
        RuntimeAdmissionSnapshot(active_jobs=1, in_flight_jobs=1),
        AdmissionBudget(max_active_jobs=4, max_in_flight_jobs=2),
    )


def test_transcription_admission_rejects_active_job_budget():
    with pytest.raises(AdmissionRejectedError) as exc_info:
        ensure_transcription_admitted(
            RuntimeAdmissionSnapshot(active_jobs=4, in_flight_jobs=0),
            AdmissionBudget(max_active_jobs=4, max_in_flight_jobs=2),
        )

    assert exc_info.value.reason == "active_job_budget_exceeded"
    assert "active job budget" in str(exc_info.value)


def test_transcription_admission_rejects_in_flight_budget():
    with pytest.raises(AdmissionRejectedError) as exc_info:
        ensure_transcription_admitted(
            RuntimeAdmissionSnapshot(active_jobs=1, in_flight_jobs=2),
            AdmissionBudget(max_active_jobs=4, max_in_flight_jobs=2),
        )

    assert exc_info.value.reason == "in_flight_job_budget_exceeded"
    assert "in-flight job budget" in str(exc_info.value)


def test_zero_or_negative_budget_disables_that_budget():
    ensure_transcription_admitted(
        RuntimeAdmissionSnapshot(active_jobs=100, in_flight_jobs=100),
        AdmissionBudget(max_active_jobs=0, max_in_flight_jobs=-1),
    )


def test_reserve_transcription_admission_uses_atomic_runtime_slot(monkeypatch):
    monkeypatch.setattr(job_runtime, "_active_job_ids", set())
    budget = AdmissionBudget(max_active_jobs=1, max_in_flight_jobs=1)

    reserve_transcription_admission("tr_one", budget)
    with pytest.raises(AdmissionRejectedError) as exc_info:
        reserve_transcription_admission("tr_two", budget)

    assert exc_info.value.reason == "active_job_budget_exceeded"
    assert job_runtime.active_job_count() == 1
    assert release_transcription_admission("tr_one") is True
    reserve_transcription_admission("tr_two", budget)


def test_in_flight_admission_returns_duplicate_before_budget_rejection(monkeypatch):
    monkeypatch.setattr(job_runtime, "_in_flight_hashes", {"sha256:one": "tr_one"})
    budget = AdmissionBudget(max_active_jobs=1, max_in_flight_jobs=1)

    assert find_in_flight_transcription("sha256:one") == "tr_one"
    duplicate = admit_transcription_in_flight("sha256:one", "tr_two", budget)
    with pytest.raises(AdmissionRejectedError) as exc_info:
        admit_transcription_in_flight("sha256:two", "tr_two", budget)

    assert duplicate.existing_job_id == "tr_one"
    assert duplicate.registered is False
    assert exc_info.value.reason == "in_flight_job_budget_exceeded"
