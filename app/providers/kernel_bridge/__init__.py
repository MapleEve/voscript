"""Python bridge for optional native provider kernels."""

from .runtime import (
    RUST_KERNEL_MODE_OFF,
    RUST_KERNEL_MODE_REQUIRED,
    RustKernelBridgeError,
    artifact_manifest_contract,
    core_smoke,
    postprocess_segments,
    require_rust_core,
    rust_kernel_mode,
    rust_provider_paths_enabled,
    status_payload_contract,
    voiceprint_score,
)
from .release_gates import (
    REQUIRED_CI_GATES,
    REQUIRED_HARD_FAIL_MODES,
    RUST_KERNEL_MODE_ROLLBACK,
    RustKernelReleaseGate,
    release_gate_matrix,
    validate_release_gate_matrix,
)

__all__ = [
    "REQUIRED_CI_GATES",
    "REQUIRED_HARD_FAIL_MODES",
    "RUST_KERNEL_MODE_OFF",
    "RUST_KERNEL_MODE_REQUIRED",
    "RUST_KERNEL_MODE_ROLLBACK",
    "RustKernelBridgeError",
    "RustKernelReleaseGate",
    "artifact_manifest_contract",
    "core_smoke",
    "postprocess_segments",
    "release_gate_matrix",
    "require_rust_core",
    "rust_kernel_mode",
    "rust_provider_paths_enabled",
    "status_payload_contract",
    "validate_release_gate_matrix",
    "voiceprint_score",
]
