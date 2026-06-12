"""Application-level runtime admission policy for transcription jobs."""

from __future__ import annotations

from dataclasses import dataclass

from infra.job_runtime import (
    active_job_count,
    in_flight_count,
    lookup_in_flight,
    release_active_job,
    try_register_in_flight,
    try_reserve_active_job,
)


@dataclass(frozen=True)
class AdmissionBudget:
    max_active_jobs: int
    max_in_flight_jobs: int


@dataclass(frozen=True)
class RuntimeAdmissionSnapshot:
    active_jobs: int
    in_flight_jobs: int


@dataclass(frozen=True)
class InFlightAdmission:
    existing_job_id: str | None = None
    registered: bool = False


class AdmissionRejectedError(RuntimeError):
    """Raised when a transcription job exceeds configured runtime budgets."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def _budget_enabled(value: int) -> bool:
    return value > 0


def ensure_transcription_admitted(
    snapshot: RuntimeAdmissionSnapshot,
    budget: AdmissionBudget,
) -> None:
    """Reject new transcription work before background execution starts."""

    if (
        _budget_enabled(budget.max_active_jobs)
        and snapshot.active_jobs >= budget.max_active_jobs
    ):
        raise AdmissionRejectedError(
            "active_job_budget_exceeded",
            (
                "Transcription active job budget exceeded "
                f"({snapshot.active_jobs}/{budget.max_active_jobs})"
            ),
        )
    if (
        _budget_enabled(budget.max_in_flight_jobs)
        and snapshot.in_flight_jobs >= budget.max_in_flight_jobs
    ):
        raise AdmissionRejectedError(
            "in_flight_job_budget_exceeded",
            (
                "Transcription in-flight job budget exceeded "
                f"({snapshot.in_flight_jobs}/{budget.max_in_flight_jobs})"
            ),
        )


def find_in_flight_transcription(file_hash: str) -> str | None:
    return lookup_in_flight(file_hash)


def reserve_transcription_admission(
    job_id: str,
    budget: AdmissionBudget,
) -> None:
    reservation = try_reserve_active_job(
        job_id,
        max_entries=budget.max_active_jobs,
    )
    if reservation.budget_exceeded:
        raise AdmissionRejectedError(
            "active_job_budget_exceeded",
            (
                "Transcription active job budget exceeded "
                f"({active_job_count()}/{budget.max_active_jobs})"
            ),
        )


def release_transcription_admission(job_id: str) -> bool:
    return release_active_job(job_id)


def admit_transcription_in_flight(
    file_hash: str,
    job_id: str,
    budget: AdmissionBudget,
) -> InFlightAdmission:
    registration = try_register_in_flight(
        file_hash,
        job_id,
        max_entries=budget.max_in_flight_jobs,
    )
    if registration.existing_job_id:
        return InFlightAdmission(existing_job_id=registration.existing_job_id)
    if registration.budget_exceeded:
        raise AdmissionRejectedError(
            "in_flight_job_budget_exceeded",
            (
                "Transcription in-flight job budget exceeded "
                f"({in_flight_count()}/{budget.max_in_flight_jobs})"
            ),
        )
    return InFlightAdmission(registered=registration.registered)
