"""Regression tests for public configuration defaults."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _fresh_config(
    monkeypatch,
    model_idle_timeout_sec: str | None = None,
    rust_kernel_mode: str | None = None,
):
    if model_idle_timeout_sec is None:
        monkeypatch.delenv("MODEL_IDLE_TIMEOUT_SEC", raising=False)
    else:
        monkeypatch.setenv("MODEL_IDLE_TIMEOUT_SEC", model_idle_timeout_sec)

    if rust_kernel_mode is None:
        monkeypatch.delenv("RUST_KERNEL_MODE", raising=False)
    else:
        monkeypatch.setenv("RUST_KERNEL_MODE", rust_kernel_mode)

    sys.modules.pop("config", None)
    return importlib.import_module("config")


def test_model_idle_timeout_defaults_to_three_minutes(monkeypatch):
    config = _fresh_config(monkeypatch)

    assert config.MODEL_IDLE_TIMEOUT_SEC == 180.0


def test_model_idle_timeout_explicit_zero_disables_idle_unload(monkeypatch):
    config = _fresh_config(monkeypatch, model_idle_timeout_sec="0")

    assert config.MODEL_IDLE_TIMEOUT_SEC == 0.0


def test_rust_kernel_mode_defaults_to_required(monkeypatch):
    config = _fresh_config(monkeypatch)

    assert config.RUST_KERNEL_MODE == "required"


def test_rust_kernel_mode_is_normalized(monkeypatch):
    config = _fresh_config(monkeypatch, rust_kernel_mode=" REQUIRED ")

    assert config.RUST_KERNEL_MODE == "required"


def test_rust_kernel_mode_explicit_off_remains_config_rollback(monkeypatch):
    config = _fresh_config(monkeypatch, rust_kernel_mode=" off ")

    assert config.RUST_KERNEL_MODE == "off"


def test_compose_default_requires_rust_kernel():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")

    assert "RUST_KERNEL_MODE=${RUST_KERNEL_MODE:-required}" in compose
    assert "RUST_KERNEL_MODE=${RUST_KERNEL_MODE:-off}" not in compose


def test_public_docs_describe_required_rust_kernel_default():
    docs = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in (
            "doc/configuration.en.md",
            "doc/configuration.zh.md",
            "doc/changelog.en.md",
            "doc/changelog.zh.md",
        )
    )
    dflt = "def" + "ault"
    py_word = "Py" + "thon"

    for stale_phrase in (
        "The " + dflt + " `off`",
        dflt + " remains " + py_word,
        "`RUST_KERNEL_MODE` | " + "`off`",
        "默认 " + "`off`",
        "默认仍使用 " + py_word,
        "默认仍由 " + py_word,
        "默认" + "关闭时",
    ):
        assert stale_phrase not in docs

    assert "`RUST_KERNEL_MODE` | `required`" in docs
