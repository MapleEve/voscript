"""Tests for public docs/code drift guardrails."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DOCS_CODE_DRIFT_GATE = ROOT / "voscript-api" / "scripts" / "docs_code_drift_gate.py"
GATE_SURFACE_FILES = (
    ".env.example",
    "README.md",
    "README.en.md",
    "app/config.py",
    "app/main.py",
    "app/api/routers/health.py",
    "app/api/routers/transcriptions.py",
    "app/api/routers/voiceprints.py",
    "doc/api.zh.md",
    "doc/api.en.md",
    "doc/configuration.zh.md",
    "doc/configuration.en.md",
    "docker-compose.yml",
)


def _load_docs_code_drift_gate():
    spec = importlib.util.spec_from_file_location(
        "voscript_docs_code_drift_gate",
        DOCS_CODE_DRIFT_GATE,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _copy_gate_surface(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    for rel_path in GATE_SURFACE_FILES:
        source = ROOT / rel_path
        target = repo_root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return repo_root


def _has_finding(
    report: dict,
    *,
    category: str,
    path: str,
    term: str,
) -> bool:
    return any(
        finding["category"] == category
        and finding["path"] == path
        and finding["term"] == term
        for finding in report["findings"]
    )


def test_docs_code_drift_gate_has_no_findings():
    gate = _load_docs_code_drift_gate()
    report = gate.build_report(ROOT)

    assert report["findings"] == []
    assert any(
        route["method"] == "GET"
        and route["path"] == "/api/transcriptions/{tr_id}/audio"
        for route in report["checked_routes"]
    )
    assert any(
        route["method"] == "GET" and route["path"] == "/api/voiceprints/{speaker_id}"
        for route in report["checked_routes"]
    )
    assert "WHISPERX_ALIGN_DEVICE" in report["public_config_keys"]


def test_reports_existing_api_route_missing_from_docs(tmp_path: Path):
    gate = _load_docs_code_drift_gate()
    root = _copy_gate_surface(tmp_path)
    route_term = "GET /api/transcriptions/{tr_id}/audio"
    assert gate.build_report(root)["findings"] == []

    for doc_path in ("doc/api.zh.md", "doc/api.en.md"):
        path = root / doc_path
        text = path.read_text(encoding="utf-8")
        assert route_term in text
        path.write_text(
            text.replace(route_term, "GET /api/transcriptions/{tr_id}/download"),
            encoding="utf-8",
        )

    report = gate.build_report(root)

    assert _has_finding(
        report,
        category="api_route_missing_from_docs",
        path="doc/api.zh.md",
        term=route_term,
    ), report["findings"]
    assert _has_finding(
        report,
        category="api_route_missing_from_docs",
        path="doc/api.en.md",
        term=route_term,
    ), report["findings"]


def test_reports_env_example_key_missing_from_compose(tmp_path: Path):
    gate = _load_docs_code_drift_gate()
    root = _copy_gate_surface(tmp_path)
    env_key = "WHISPERX_ALIGN_DEVICE"
    compose_path = root / "docker-compose.yml"
    assert gate.build_report(root)["findings"] == []

    text = compose_path.read_text(encoding="utf-8")
    compose_ref = "      - WHISPERX_ALIGN_DEVICE=${WHISPERX_ALIGN_DEVICE:-cpu}\n"
    assert compose_ref in text
    compose_path.write_text(
        text.replace(compose_ref, "      - WHISPERX_ALIGN_DEVICE=cpu\n"),
        encoding="utf-8",
    )

    report = gate.build_report(root)

    assert _has_finding(
        report,
        category="env_example_key_missing_from_compose",
        path="docker-compose.yml",
        term=env_key,
    ), report["findings"]


def test_reports_route_added_to_main_without_docs(tmp_path: Path):
    gate = _load_docs_code_drift_gate()
    root = _copy_gate_surface(tmp_path)
    main_path = root / "app/main.py"
    new_router_path = root / "app/api/routers/drift_probe.py"
    route_term = "GET /api/drift-probe"
    assert gate.build_report(root)["findings"] == []

    main_text = main_path.read_text(encoding="utf-8")
    assert "from api.routers import health, transcriptions, voiceprints" in main_text
    main_text = main_text.replace(
        "from api.routers import health, transcriptions, voiceprints",
        "from api.routers import health, transcriptions, voiceprints, drift_probe",
    )
    main_text = main_text.replace(
        "app.include_router(voiceprints.router)\n",
        "app.include_router(voiceprints.router)\n"
        "app.include_router(drift_probe.router)\n",
    )
    main_path.write_text(main_text, encoding="utf-8")
    new_router_path.write_text(
        '"""Router used by the docs/code drift gate regression test."""\n\n'
        "from fastapi import APIRouter\n\n\n"
        'router = APIRouter(prefix="/api")\n\n\n'
        '@router.get("/drift-probe")\n'
        "async def drift_probe():\n"
        '    return {"ok": True}\n',
        encoding="utf-8",
    )

    report = gate.build_report(root)

    assert any(
        route["method"] == "GET"
        and route["path"] == "/api/drift-probe"
        and route["source"] == "app/api/routers/drift_probe.py"
        for route in report["checked_routes"]
    )
    assert _has_finding(
        report,
        category="api_route_missing_from_docs",
        path="doc/api.zh.md",
        term=route_term,
    ), report["findings"]
    assert _has_finding(
        report,
        category="api_route_missing_from_docs",
        path="doc/api.en.md",
        term=route_term,
    ), report["findings"]
