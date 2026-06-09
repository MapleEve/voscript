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

__all__ = [
    "RUST_KERNEL_MODE_OFF",
    "RUST_KERNEL_MODE_REQUIRED",
    "RustKernelBridgeError",
    "artifact_manifest_contract",
    "core_smoke",
    "postprocess_segments",
    "require_rust_core",
    "rust_kernel_mode",
    "rust_provider_paths_enabled",
    "status_payload_contract",
    "voiceprint_score",
]
