"""Tests for static provider capability metadata and language matching."""

from __future__ import annotations

import pytest

import providers.capabilities as capabilities_module
from providers.capabilities import (
    ProviderCapability,
    ProviderCapabilityError,
    default_provider_capabilities,
    get_provider_capability,
    match_provider_capability,
)


def test_default_asr_capability_is_multilingual_required_stage():
    capability = get_provider_capability("asr")

    assert capability.stage == "asr"
    assert capability.stage_criticality == "required"
    assert "*" in capability.supported_languages

    match = match_provider_capability("asr", language="zh")

    assert match.should_run is True
    assert match.reason == "language_supported"
    assert match.metadata["language"] == "zh"


def test_required_stage_language_mismatch_hard_fails(monkeypatch):
    custom = ProviderCapability(
        stage="asr",
        name="default",
        supported_languages=frozenset({"en"}),
    )
    monkeypatch.setitem(
        capabilities_module._DEFAULT_CAPABILITIES,
        ("asr", "default"),
        custom,
    )

    with pytest.raises(ProviderCapabilityError, match="Required stage"):
        match_provider_capability("asr", language="zh")


def test_degradable_stage_language_mismatch_skips_with_safe_metadata(monkeypatch):
    capability = get_provider_capability("alignment")
    monkeypatch.setitem(
        capabilities_module._DEFAULT_CAPABILITIES,
        ("alignment", "default"),
        ProviderCapability(
            stage=capability.stage,
            name=capability.name,
            supported_languages=capability.supported_languages,
            disabled_languages=frozenset({"zh"}),
            stage_criticality="degradable",
            failure_policy="skip",
        ),
    )

    match = match_provider_capability("alignment", language="zh")

    assert match.should_run is False
    assert match.reason == "language_disabled"
    assert match.metadata == {
        "stage": "alignment",
        "provider": "default",
        "criticality": "degradable",
        "language": "zh",
        "reason": "language_disabled",
        "action": "skip",
    }


def test_unknown_provider_capability_hard_fails():
    with pytest.raises(ProviderCapabilityError, match="No provider capability"):
        get_provider_capability("asr", "missing")


def test_default_provider_capabilities_are_static_records():
    capabilities = default_provider_capabilities()

    assert capabilities
    assert all(isinstance(capability, ProviderCapability) for capability in capabilities)
