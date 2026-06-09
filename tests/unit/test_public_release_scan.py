"""Tests for the repository-owned public release scanner."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCANNER = ROOT / "voscript-api" / "scripts" / "public_release_scan.py"


def _run_git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, stdout=subprocess.PIPE)


def _scan_fixture(content: str) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _run_git(root, "init", "-q")
        fixture = root / "fixture.md"
        fixture.write_text(content, encoding="utf-8")
        _run_git(root, "add", "fixture.md")
        return subprocess.run(
            [sys.executable, str(SCANNER), "--root", str(root)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )


def test_public_release_scan_allows_placeholders():
    result = _scan_fixture(
        "\n".join(
            [
                "Authorization: Bearer <API_KEY>",
                "HF_TOKEN=${HF_TOKEN}",
                "API_KEY=your-api-key",
                "Use internal live validation for release notes.",
            ]
        )
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Public release scan passed" in result.stdout


def test_public_release_scan_blocks_secret_looking_assignments():
    synthetic_secret = "sk-live-" + "syntheticsecret123456789"

    result = _scan_fixture(f"API_KEY={synthetic_secret}")

    assert result.returncode == 1
    assert "secret-looking assignment" in result.stdout


def test_public_release_scan_blocks_private_paths_and_real_ids():
    local_path = "/" + "Users/example/private.log"
    transcription_id = "tr_" + "20260426_124218_abcdef"
    speaker_id = "spk_" + "1234abcd"
    fixture = "\n".join(
        [
            f"Read {local_path} before publishing.",
            f"Result id {transcription_id} should not be public.",
            f"Speaker id {speaker_id} should not be public.",
        ]
    )

    result = _scan_fixture(fixture)

    assert result.returncode == 1
    assert "machine-local path" in result.stdout
    assert "real transcription id" in result.stdout
    assert "real speaker id" in result.stdout
