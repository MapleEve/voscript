#!/usr/bin/env python3
"""Check public docs against VoScript's runtime surface anchors."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

CONFIG_DOC_FILES = (
    "doc/configuration.zh.md",
    "doc/configuration.en.md",
    ".env.example",
)
API_DOC_FILES = (
    "doc/api.zh.md",
    "doc/api.en.md",
)
README_FILES = (
    "README.md",
    "README.en.md",
)
ENV_HELPERS = {
    "_env_float",
    "_env_int",
    "_env_str",
    "_env_csv_set",
    "_env_mapping",
}
PLACEHOLDER_ENV_VALUES = (
    "",
    "change-me-to-a-long-random-string",
)
PLACEHOLDER_ENV_MARKERS = (
    "placeholder",
    "replace-me",
    "replace_me",
)

RESULT_CONTRACT_TERMS = (
    "status",
    "segments[].speaker_label",
    "segments[].words",
    "alignment",
    "artifacts",
    "speaker_map",
    "unique_speakers",
    "similarity",
    "params",
    "no_repeat_ngram_size",
    "MAX_UPLOAD_BYTES",
)

RUST_MODE_TERMS = (
    "RUST_KERNEL_MODE",
    "off",
    "required",
    "fail closed",
)


def _read(root: Path, rel_path: str) -> str:
    return (root / rel_path).read_text(encoding="utf-8")


def _finding(category: str, path: str, term: str, advice: str) -> dict[str, str]:
    return {
        "category": category,
        "path": path,
        "term": term,
        "advice": advice,
    }


def _string_arg(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _literal_value(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp):
        value = _literal_value(node.operand)
        if isinstance(node.op, ast.USub) and isinstance(value, (int, float)):
            return -value
    if isinstance(node, ast.BinOp):
        left = _literal_value(node.left)
        right = _literal_value(node.right)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
        value = _literal_value(node.args[0])
        if node.func.id == "str":
            return str(value)
        if node.func.id == "int":
            return int(value)
        if node.func.id == "float":
            return float(value)
    return None


def _default_terms(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, float) and value.is_integer():
        return (str(int(value)), str(value))
    return (str(value),)


def config_env_defaults(root: Path) -> dict[str, tuple[str, ...]]:
    tree = ast.parse(_read(root, "app/config.py"), filename="app/config.py")
    defaults: dict[str, tuple[str, ...]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            if node.func.attr != "getenv" or len(node.args) < 1:
                continue
            key = _string_arg(node.args[0])
            if key is None:
                continue
            default = _literal_value(node.args[1]) if len(node.args) > 1 else None
            defaults.setdefault(key, _default_terms(default))
            continue
        if not isinstance(node.func, ast.Name) or node.func.id not in ENV_HELPERS:
            continue
        if not node.args:
            continue
        key = _string_arg(node.args[0])
        if key is None:
            continue
        default = _literal_value(node.args[1]) if len(node.args) > 1 else None
        defaults.setdefault(key, _default_terms(default))
    return dict(sorted(defaults.items()))


def env_example_values(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in _read(root, ".env.example").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key:
            values[key] = value
    return dict(sorted(values.items()))


def compose_variable_refs(root: Path) -> set[str]:
    import re

    compose = _read(root, "docker-compose.yml")
    return set(re.findall(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-[^}]*)?\}", compose))


def router_sources(root: Path) -> list[tuple[str, str]]:
    main_tree = ast.parse(_read(root, "app/main.py"), filename="app/main.py")
    imported_routers: dict[str, str] = {}
    included_router_names: set[str] = set()
    for node in ast.walk(main_tree):
        if isinstance(node, ast.ImportFrom) and node.module == "api.routers":
            for alias in node.names:
                imported_routers[alias.asname or alias.name] = alias.name
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute) or func.attr != "include_router":
            continue
        if not node.args:
            continue
        first_arg = node.args[0]
        if (
            isinstance(first_arg, ast.Attribute)
            and first_arg.attr == "router"
            and isinstance(first_arg.value, ast.Name)
        ):
            included_router_names.add(first_arg.value.id)

    sources: list[tuple[str, str]] = []
    for local_name in sorted(included_router_names):
        router_module = imported_routers.get(local_name)
        if router_module is None:
            continue
        rel_path = f"app/api/routers/{router_module}.py"
        sources.append((rel_path, router_prefix(root, rel_path)))
    return sources


def router_prefix(root: Path, rel_path: str) -> str:
    tree = ast.parse(_read(root, rel_path), filename=rel_path)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Name) or node.func.id != "APIRouter":
            continue
        for keyword in node.keywords:
            if keyword.arg == "prefix":
                value = _string_arg(keyword.value)
                return value or ""
        return ""
    return ""


def public_routes(root: Path) -> list[dict[str, str]]:
    routes: list[dict[str, str]] = []
    for rel_path, prefix in router_sources(root):
        tree = ast.parse(_read(root, rel_path), filename=rel_path)
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                func = decorator.func
                if not isinstance(func, ast.Attribute):
                    continue
                if func.attr not in {"get", "post", "put", "delete"}:
                    continue
                if not isinstance(func.value, ast.Name) or func.value.id != "router":
                    continue
                if not decorator.args:
                    continue
                route_path = _string_arg(decorator.args[0])
                if route_path is None:
                    continue
                routes.append(
                    {
                        "method": func.attr.upper(),
                        "path": f"{prefix}{route_path}",
                        "source": rel_path,
                        "handler": node.name,
                    }
                )
    return sorted(routes, key=lambda item: (item["path"], item["method"]))


def _route_doc_terms(route: dict[str, str]) -> tuple[str, ...]:
    method = route["method"]
    path = route["path"]
    terms = {f"{method} {path}"}
    id_alias = path
    for placeholder in ("job_id", "speaker_id"):
        id_alias = id_alias.replace("{" + placeholder + "}", "{id}")
    terms.add(f"{method} {id_alias}")
    return tuple(sorted(terms))


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _is_placeholder_env_value(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in PLACEHOLDER_ENV_VALUES or any(
        marker in normalized for marker in PLACEHOLDER_ENV_MARKERS
    )


def _check_public_routes(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    routes = public_routes(root)
    api_docs = {path: _read(root, path) for path in API_DOC_FILES}
    for route in routes:
        terms = _route_doc_terms(route)
        for doc_path, text in api_docs.items():
            if not _contains_any(text, terms):
                findings.append(
                    _finding(
                        "api_route_missing_from_docs",
                        doc_path,
                        " or ".join(terms),
                        f"Document route from {route['source']}::{route['handler']}.",
                    )
                )
    return findings


def _check_config_docs(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    config_docs = {path: _read(root, path) for path in CONFIG_DOC_FILES}
    config_defaults = config_env_defaults(root)
    env_values = env_example_values(root)
    compose_refs = compose_variable_refs(root)
    for key in sorted(config_defaults):
        for doc_path in CONFIG_DOC_FILES[:2]:
            text = config_docs[doc_path]
            if key not in text:
                findings.append(
                    _finding(
                        "config_env_key_missing_from_config_docs",
                        doc_path,
                        key,
                        "Document config.py env keys or state why they are not public knobs.",
                    )
                )

    for key in sorted(env_values):
        for doc_path, text in config_docs.items():
            if key not in text:
                findings.append(
                    _finding(
                        "public_config_key_missing_from_docs",
                        doc_path,
                        key,
                        "Keep public env/config docs in sync with config.py and compose.",
                    )
                )

    for key in sorted(env_values):
        if key not in compose_refs:
            findings.append(
                _finding(
                    "env_example_key_missing_from_compose",
                    "docker-compose.yml",
                    key,
                    "Keep .env.example keys wired into compose or remove the public knob.",
                )
            )

    for key, defaults in config_defaults.items():
        if not defaults:
            continue
        for doc_path, text in config_docs.items():
            if not any(default in text for default in defaults):
                findings.append(
                    _finding(
                        "public_config_default_missing_from_docs",
                        doc_path,
                        f"{key}={'/'.join(defaults)}",
                        "Document public defaults in both languages and .env.example.",
                    )
                )

    for key, value in env_values.items():
        if _is_placeholder_env_value(value):
            continue
        for doc_path in CONFIG_DOC_FILES[:2]:
            text = config_docs[doc_path]
            if value and value not in text:
                findings.append(
                    _finding(
                        "env_example_default_missing_from_config_docs",
                        doc_path,
                        f"{key}={value}",
                        "Keep .env.example defaults aligned with configuration docs.",
                    )
                )
    return findings


def _check_contract_docs(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    api_docs = {path: _read(root, path) for path in API_DOC_FILES}
    for term in RESULT_CONTRACT_TERMS:
        for doc_path, text in api_docs.items():
            if term not in text:
                findings.append(
                    _finding(
                        "result_contract_term_missing_from_api_docs",
                        doc_path,
                        term,
                        "Keep public result/status/artifact contract docs synchronized.",
                    )
                )

    config_docs = {path: _read(root, path) for path in CONFIG_DOC_FILES[:2]}
    for term in RUST_MODE_TERMS:
        for doc_path, text in config_docs.items():
            if term not in text:
                findings.append(
                    _finding(
                        "rust_mode_term_missing_from_config_docs",
                        doc_path,
                        term,
                        "Keep Rust mode wording precise: required by default and fail closed; off is an explicit rollback.",
                    )
                )
    return findings


def _check_readme_links(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for path in README_FILES:
        text = _read(root, path)
        for term in ("configuration.", "api."):
            if term not in text:
                findings.append(
                    _finding(
                        "readme_public_doc_link_missing",
                        path,
                        term,
                        "README must route users to configuration and API references.",
                    )
                )
    return findings


def build_report(root: Path) -> dict[str, Any]:
    config_defaults = config_env_defaults(root)
    env_values = env_example_values(root)
    findings: list[dict[str, str]] = []
    findings.extend(_check_public_routes(root))
    findings.extend(_check_config_docs(root))
    findings.extend(_check_contract_docs(root))
    findings.extend(_check_readme_links(root))
    return {
        "api_docs": list(API_DOC_FILES),
        "checked_routes": public_routes(root),
        "compose_variable_refs": sorted(compose_variable_refs(root)),
        "config_env_defaults": {
            key: list(defaults) for key, defaults in config_defaults.items()
        },
        "config_docs": list(CONFIG_DOC_FILES),
        "env_example_keys": sorted(env_values),
        "public_config_keys": sorted(set(config_defaults) | set(env_values)),
        "router_sources": [
            {"path": path, "prefix": prefix} for path, prefix in router_sources(root)
        ],
        "findings": findings,
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"checked_routes: {len(report['checked_routes'])}")
    print(f"public_config_keys: {len(report['public_config_keys'])}")
    if report["findings"]:
        print("docs/code drift findings:")
        for item in report["findings"]:
            print(f"- {item['path']}: {item['category']}: {item['term']}")
            print(f"  {item['advice']}")
    else:
        print("docs/code drift gate passed")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when docs/code drift findings are present.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    report = build_report(root)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text(report)

    if args.check and report["findings"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
