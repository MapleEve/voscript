"""Internal release-gate matrix for selected Rust-backed provider paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

RUST_KERNEL_MODE_ROLLBACK: Final = "RUST_KERNEL_MODE=off"

HARD_FAIL_IMPORT_FAILURE: Final = "import_failure"
HARD_FAIL_CALL_FAILURE: Final = "call_failure"
HARD_FAIL_INVALID_PAYLOAD: Final = "invalid_payload"
HARD_FAIL_INVALID_RESPONSE: Final = "invalid_response"
HARD_FAIL_PARITY_MISMATCH: Final = "parity_mismatch"

REQUIRED_HARD_FAIL_MODES: Final = frozenset(
    {
        HARD_FAIL_IMPORT_FAILURE,
        HARD_FAIL_CALL_FAILURE,
        HARD_FAIL_INVALID_PAYLOAD,
        HARD_FAIL_INVALID_RESPONSE,
        HARD_FAIL_PARITY_MISMATCH,
    }
)

REQUIRED_CI_GATES: Final = frozenset(
    {
        "python_unit_security_tests",
        "kernel_bridge_smoke_tests",
        "rust_fmt",
        "rust_clippy",
        "rust_tests",
        "rust_wheel_smoke",
        "docker_packaging_smoke",
        "public_release_scan",
    }
)


@dataclass(frozen=True, slots=True)
class RustKernelReleaseGate:
    """Audit contract for one selected Rust-backed implementation path.

    The runtime switch stays intentionally small: Python owns orchestration and
    public API shape; Rust owns only selected pure kernels/helpers. This matrix
    makes the selected paths and their rollback/fail-closed evidence explicit.
    """

    name: str
    bridge_function: str
    python_owner: str
    rust_owner: str
    rollback: str
    regression_matrix: tuple[str, ...]
    hard_fail_modes: frozenset[str]
    ci_gates: frozenset[str]
    performance_baseline: str
    public_api_change: bool = False


SELECTED_RUST_KERNEL_GATES: Final = (
    RustKernelReleaseGate(
        name="voiceprint_scoring",
        bridge_function="voiceprint_score",
        python_owner="voiceprints.scoring.score_voiceprint_candidates",
        rust_owner="voscript_core::voiceprint::score_voiceprint_candidates",
        rollback=RUST_KERNEL_MODE_ROLLBACK,
        regression_matrix=(
            "raw_cosine_adaptive_threshold",
            "asnorm_active_with_margin_guard",
            "small_cohort_raw_fallback",
            "non_finite_embedding_rejection",
            "db_required_mode_payload_export",
            "db_off_mode_python_scoring",
        ),
        hard_fail_modes=REQUIRED_HARD_FAIL_MODES,
        ci_gates=REQUIRED_CI_GATES,
        performance_baseline="internal_scoring_only_synthetic",
    ),
    RustKernelReleaseGate(
        name="postprocess_segments",
        bridge_function="postprocess_segments",
        python_owner="postprocess.segments.build_result_segments",
        rust_owner="voscript_core::postprocess::build_result_segments",
        rollback=RUST_KERNEL_MODE_ROLLBACK,
        regression_matrix=(
            "word_normalization",
            "adjacent_text_segment_merge",
            "word_payload_merge_block",
            "duplicate_display_name_disambiguation",
            "stable_speaker_label_preservation",
        ),
        hard_fail_modes=REQUIRED_HARD_FAIL_MODES,
        ci_gates=REQUIRED_CI_GATES,
        performance_baseline="internal_postprocess_only_synthetic",
    ),
    RustKernelReleaseGate(
        name="artifact_manifest_contract",
        bridge_function="artifact_manifest_contract",
        python_owner="pipeline.contracts.artifacts.build_artifact_manifest",
        rust_owner="voscript_core::contracts::artifact_manifest_contract",
        rollback=RUST_KERNEL_MODE_ROLLBACK,
        regression_matrix=(
            "public_safe_manifest_build",
            "path_and_url_rejection",
            "legacy_unknown_entry_tolerance",
            "stable_optional_experimental_categories",
        ),
        hard_fail_modes=REQUIRED_HARD_FAIL_MODES,
        ci_gates=REQUIRED_CI_GATES,
        performance_baseline="internal_helper_only_synthetic",
    ),
    RustKernelReleaseGate(
        name="status_payload_contract",
        bridge_function="status_payload_contract",
        python_owner="pipeline.contracts.status.build_status_payload",
        rust_owner="voscript_core::contracts::status_payload_contract",
        rollback=RUST_KERNEL_MODE_ROLLBACK,
        regression_matrix=(
            "known_status_normalization",
            "unknown_legacy_status_to_failed",
            "basename_only_filename",
            "legacy_status_payload_compatibility",
        ),
        hard_fail_modes=REQUIRED_HARD_FAIL_MODES,
        ci_gates=REQUIRED_CI_GATES,
        performance_baseline="internal_helper_only_synthetic",
    ),
)


def release_gate_matrix() -> tuple[RustKernelReleaseGate, ...]:
    """Return the selected Rust-backed release gates as an immutable tuple."""

    return SELECTED_RUST_KERNEL_GATES


def validate_release_gate_matrix(
    gates: tuple[RustKernelReleaseGate, ...] = SELECTED_RUST_KERNEL_GATES,
) -> tuple[str, ...]:
    """Return policy gaps that would block a 0.8.x Rust-backed release."""

    gaps: list[str] = []
    names: set[str] = set()
    bridge_functions: set[str] = set()
    for gate in gates:
        if gate.name in names:
            gaps.append(f"{gate.name}: duplicate gate name")
        names.add(gate.name)
        if gate.bridge_function in bridge_functions:
            gaps.append(f"{gate.name}: duplicate bridge function")
        bridge_functions.add(gate.bridge_function)
        if not gate.regression_matrix:
            gaps.append(f"{gate.name}: missing regression matrix")
        missing_hard_fail = REQUIRED_HARD_FAIL_MODES.difference(gate.hard_fail_modes)
        if missing_hard_fail:
            gaps.append(
                f"{gate.name}: missing hard-fail modes {sorted(missing_hard_fail)}"
            )
        missing_ci = REQUIRED_CI_GATES.difference(gate.ci_gates)
        if missing_ci:
            gaps.append(f"{gate.name}: missing CI gates {sorted(missing_ci)}")
        if gate.rollback != RUST_KERNEL_MODE_ROLLBACK:
            gaps.append(f"{gate.name}: rollback must be {RUST_KERNEL_MODE_ROLLBACK}")
        if gate.public_api_change:
            gaps.append(f"{gate.name}: public API change is not allowed in 0.8.4")
    return tuple(gaps)


__all__ = [
    "HARD_FAIL_CALL_FAILURE",
    "HARD_FAIL_IMPORT_FAILURE",
    "HARD_FAIL_INVALID_PAYLOAD",
    "HARD_FAIL_INVALID_RESPONSE",
    "HARD_FAIL_PARITY_MISMATCH",
    "REQUIRED_CI_GATES",
    "REQUIRED_HARD_FAIL_MODES",
    "RUST_KERNEL_MODE_ROLLBACK",
    "RustKernelReleaseGate",
    "SELECTED_RUST_KERNEL_GATES",
    "release_gate_matrix",
    "validate_release_gate_matrix",
]
