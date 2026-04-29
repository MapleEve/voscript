"""Regression tests for public configuration defaults."""

from __future__ import annotations

import importlib
import sys


def _fresh_config(monkeypatch, value: str | None = None):
    if value is None:
        monkeypatch.delenv("MODEL_IDLE_TIMEOUT_SEC", raising=False)
    else:
        monkeypatch.setenv("MODEL_IDLE_TIMEOUT_SEC", value)

    sys.modules.pop("config", None)
    return importlib.import_module("config")


def test_model_idle_timeout_defaults_to_three_minutes(monkeypatch):
    config = _fresh_config(monkeypatch)

    assert config.MODEL_IDLE_TIMEOUT_SEC == 180.0


def test_model_idle_timeout_explicit_zero_disables_idle_unload(monkeypatch):
    config = _fresh_config(monkeypatch, "0")

    assert config.MODEL_IDLE_TIMEOUT_SEC == 0.0
