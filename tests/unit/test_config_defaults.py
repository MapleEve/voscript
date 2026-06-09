"""Regression tests for public configuration defaults."""

from __future__ import annotations

import importlib
import sys


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


def test_rust_kernel_mode_defaults_to_off(monkeypatch):
    config = _fresh_config(monkeypatch)

    assert config.RUST_KERNEL_MODE == "off"


def test_rust_kernel_mode_is_normalized(monkeypatch):
    config = _fresh_config(monkeypatch, rust_kernel_mode=" REQUIRED ")

    assert config.RUST_KERNEL_MODE == "required"
