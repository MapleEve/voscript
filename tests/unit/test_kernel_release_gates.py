"""Regression, rollback, and CI gate coverage for 0.8.x Rust kernels."""

from __future__ import annotations

from pathlib import Path

from providers.kernel_bridge.release_gates import (
    REQUIRED_CI_GATES,
    REQUIRED_HARD_FAIL_MODES,
    RUST_KERNEL_MODE_ROLLBACK,
    release_gate_matrix,
    validate_release_gate_matrix,
)


ROOT = Path(__file__).resolve().parents[2]


def test_release_gate_matrix_covers_selected_rust_backed_paths():
    gates = release_gate_matrix()

    assert {gate.name for gate in gates} == {
        "voiceprint_scoring",
        "postprocess_segments",
        "artifact_manifest_contract",
        "status_payload_contract",
    }
    assert {gate.bridge_function for gate in gates} == {
        "voiceprint_score",
        "postprocess_segments",
        "artifact_manifest_contract",
        "status_payload_contract",
    }


def test_release_gate_matrix_has_no_policy_gaps():
    assert validate_release_gate_matrix() == ()


def test_each_selected_gate_is_fail_closed_and_explicitly_rollbackable():
    for gate in release_gate_matrix():
        assert gate.rollback == RUST_KERNEL_MODE_ROLLBACK
        assert REQUIRED_HARD_FAIL_MODES.issubset(gate.hard_fail_modes)
        assert REQUIRED_CI_GATES.issubset(gate.ci_gates)
        assert gate.regression_matrix
        assert gate.performance_baseline.startswith("internal_")
        assert gate.public_api_change is False


def test_ci_workflows_include_required_release_gate_commands():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    heavy = (ROOT / ".github" / "workflows" / "rust-foundation-heavy.yml").read_text(
        encoding="utf-8"
    )
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(
        encoding="utf-8"
    )

    assert "public_release_scan.py --root ." in ci
    assert "architecture_gate.py --root . --check" in ci
    assert "pytest tests/unit/ tests/test_security.py" in ci
    assert (
        "cargo fmt --manifest-path crates/voscript_core/Cargo.toml -- --check" in heavy
    )
    assert "resolve-source:" in heavy
    assert "git rev-parse HEAD" in heavy
    assert "needs.resolve-source.outputs.source-sha" in heavy
    assert "cargo clippy --manifest-path crates/voscript_core/Cargo.toml" in heavy
    assert "cargo test --manifest-path crates/voscript_core/Cargo.toml" in heavy
    assert (
        "maturin build --release --manifest-path crates/voscript_core/Cargo.toml"
        in heavy
    )
    assert "docker build ./app" in heavy
    assert "RUST_KERNEL_MODE=required" in heavy
    assert "workflow_dispatch:" in heavy
    assert "types: [opened, reopened, ready_for_review]" in heavy
    assert "voscript-rust-foundation:${{ github.sha }}" not in heavy
    assert heavy.count("ref: ${{ github.event.inputs.ref || github.ref }}") == 1
    assert "resolve-source:" in release
    assert "source-sha" in release
    assert "git rev-parse HEAD" in release
    assert "ref: ${{ needs.resolve-source.outputs.source-sha }}" in release
    assert "public-release-scan" in release
    assert "lint-format" in release
    assert "unit-security" in release
    assert "docker-smoke" in release
    assert "Run container Rust extension smoke" in release
    assert "Run container healthz smoke" in release
    assert (
        "maturin build --release --manifest-path crates/voscript_core/Cargo.toml"
        in release
    )
    assert "VOSCRIPT_CORE_WHEEL" in release
    assert "sha-$SOURCE_SHA" in release
    assert (
        "org.opencontainers.image.revision=${{ needs.resolve-source.outputs.source-sha }}"
        in release
    )
    assert (
        "voscript-core-wheel-${{ needs.resolve-source.outputs.source-sha }}"
        in release
    )


def test_public_release_scan_entrypoint_is_repo_owned():
    scan = ROOT / "voscript-api" / "scripts" / "public_release_scan.py"

    assert scan.exists()
    assert "Public release scan passed" in scan.read_text(encoding="utf-8")
