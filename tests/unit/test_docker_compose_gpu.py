"""Regression tests for Docker GPU exposure defaults."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


def _load_yaml_config(text: str) -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(text)


def _normalize_environment(environment: dict | list | None) -> dict:
    if environment is None:
        return {}
    if isinstance(environment, dict):
        return environment

    normalized = {}
    for item in environment:
        key, _, value = item.partition("=")
        normalized[key] = value
    return normalized


def _render_compose_config(tmp_path: Path) -> dict:
    """Prefer real Compose rendering; fall back to structured YAML parsing."""
    if shutil.which("docker"):
        empty_env = tmp_path / "empty.env"
        empty_env.write_text("", encoding="utf-8")
        env = os.environ.copy()
        env.pop("CUDA_VISIBLE_DEVICES", None)

        result = subprocess.run(
            [
                "docker",
                "compose",
                "--env-file",
                str(empty_env),
                "-f",
                str(ROOT / "docker-compose.yml"),
                "config",
                "--format",
                "json",
            ],
            cwd=ROOT,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)

    return _load_yaml_config((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_config_defaults_to_all_visible_gpus_without_cuda_visible_devices(
    tmp_path,
):
    config = _render_compose_config(tmp_path)
    service = config["services"]["voscript"]
    environment = _normalize_environment(service.get("environment"))

    assert "CUDA_VISIBLE_DEVICES" not in environment

    devices = service["deploy"]["resources"]["reservations"]["devices"]
    assert len(devices) == 1
    assert devices[0]["driver"] == "nvidia"
    assert devices[0]["count"] in {"all", -1}
    assert devices[0]["capabilities"] == ["gpu"]


def test_env_example_does_not_default_to_gpu_zero_visibility_pin():
    env_example = (ROOT / ".env.example").read_text()

    assert "CUDA_VISIBLE_DEVICES=" not in env_example
    assert "CUDA_VISIBLE_DEVICES=0" not in env_example
