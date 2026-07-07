"""Application-level runtime admission policy for transcription jobs."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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
    min_free_disk_bytes: int = 0


class DiskUsage(Protocol):
    free: int


@dataclass(frozen=True)
class MemorySensitiveStageLimits:
    denoise_max_audio_duration_sec: float
    embedding_preload_max_audio_duration_sec: float
    whisperx_align_max_audio_duration_sec: float


@dataclass(frozen=True)
class RuntimeAdmissionSnapshot:
    active_jobs: int
    in_flight_jobs: int
    free_disk_bytes: int | None = None
    memory_sensitive_stage_limits: MemorySensitiveStageLimits | None = None
    audio_duration_seconds: float | None = None


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


def data_disk_free_bytes(
    path: Path,
    *,
    disk_usage: Callable[[Path], DiskUsage] | None = None,
) -> int:
    """Read free bytes for the data disk without binding to a web framework."""

    disk_usage = disk_usage or shutil.disk_usage
    try:
        return int(disk_usage(path).free)
    except OSError as exc:
        raise AdmissionRejectedError(
            "data_disk_pressure",
            f"Unable to inspect data disk free space for {path}",
        ) from exc


def build_runtime_admission_snapshot(
    *,
    data_path: Path | None = None,
    disk_usage: Callable[[Path], DiskUsage] | None = None,
    memory_sensitive_stage_limits: MemorySensitiveStageLimits | None = None,
    audio_duration_seconds: float | None = None,
) -> RuntimeAdmissionSnapshot:
    free_disk_bytes = None
    if data_path is not None:
        free_disk_bytes = data_disk_free_bytes(data_path, disk_usage=disk_usage)
    return RuntimeAdmissionSnapshot(
        active_jobs=active_job_count(),
        in_flight_jobs=in_flight_count(),
        free_disk_bytes=free_disk_bytes,
        memory_sensitive_stage_limits=memory_sensitive_stage_limits,
        audio_duration_seconds=audio_duration_seconds,
    )


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
    if _budget_enabled(budget.min_free_disk_bytes):
        if snapshot.free_disk_bytes is None:
            raise AdmissionRejectedError(
                "data_disk_pressure",
                "Unable to inspect data disk free space before admission",
            )
        if snapshot.free_disk_bytes < budget.min_free_disk_bytes:
            raise AdmissionRejectedError(
                "data_disk_pressure",
                (
                    "Transcription data disk free space below admission budget "
                    f"({snapshot.free_disk_bytes}/{budget.min_free_disk_bytes})"
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
