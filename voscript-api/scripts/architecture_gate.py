#!/usr/bin/env python3
"""Report and check VoScript source-level architecture gates."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

CORE_RINGS = {
    "api_composition",
    "application",
    "pipeline",
    "providers",
    "domain",
    "infra",
}


def _module_name(app_root: Path, path: Path) -> tuple[str, bool]:
    relative = path.relative_to(app_root).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        return ".".join(parts[:-1]), True
    return ".".join(parts), False


def app_modules(root: Path) -> dict[str, Path]:
    app_root = root / "app"
    modules: dict[str, Path] = {}
    for path in sorted(app_root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        module_name, _ = _module_name(app_root, path)
        modules[module_name] = path
    return modules


def _resolve_relative_import(
    current_module: str,
    is_package: bool,
    *,
    level: int,
    module: str | None,
) -> str:
    package_parts = (
        current_module.split(".") if is_package else current_module.split(".")[:-1]
    )
    prefix = package_parts[: len(package_parts) - level + 1]
    if module:
        prefix.extend(module.split("."))
    return ".".join(prefix)


def _strip_app_prefix(module_name: str) -> str:
    if module_name == "app":
        return ""
    if module_name.startswith("app."):
        return module_name.removeprefix("app.")
    return module_name


def _internal_module_for(module_name: str, modules: set[str]) -> str | None:
    candidate = _strip_app_prefix(module_name)
    if not candidate:
        return None
    parts = candidate.split(".")
    for end in range(len(parts), 0, -1):
        prefix = ".".join(parts[:end])
        if prefix in modules:
            return prefix
    return None


def _is_type_checking_guard(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    if isinstance(node, ast.Attribute):
        return node.attr == "TYPE_CHECKING"
    return False


class _ImportCollector(ast.NodeVisitor):
    def __init__(
        self,
        *,
        current_module: str,
        is_package: bool,
        modules: set[str],
    ) -> None:
        self.current_module = current_module
        self.is_package = is_package
        self.modules = modules
        self.targets: set[str] = set()
        self._type_checking_depth = 0

    def visit_If(self, node: ast.If) -> None:
        if _is_type_checking_guard(node.test):
            self._type_checking_depth += 1
            for child in node.body:
                self.visit(child)
            self._type_checking_depth -= 1
            for child in node.orelse:
                self.visit(child)
            return
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        if self._type_checking_depth:
            return
        for alias in node.names:
            self._add_internal_target(alias.name)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._type_checking_depth:
            return
        if node.level:
            base = _resolve_relative_import(
                self.current_module,
                self.is_package,
                level=node.level,
                module=node.module,
            )
        else:
            base = node.module or ""

        if base:
            self._add_internal_target(base)
        for alias in node.names:
            if alias.name == "*":
                continue
            target = f"{base}.{alias.name}" if base else alias.name
            self._add_internal_target(target)

    def _add_internal_target(self, module_name: str) -> None:
        target = _internal_module_for(module_name, self.modules)
        if target is not None and target != self.current_module:
            self.targets.add(target)


def internal_import_graph(root: Path) -> dict[str, set[str]]:
    module_paths = app_modules(root)
    modules = set(module_paths)
    app_root = root / "app"
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for module, path in module_paths.items():
        _, is_package = _module_name(app_root, path)
        collector = _ImportCollector(
            current_module=module,
            is_package=is_package,
            modules=modules,
        )
        collector.visit(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
        graph[module].update(collector.targets)
    return graph


def strongly_connected_components(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
    index = 0
    indexes: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def strongconnect(node: str) -> None:
        nonlocal index
        indexes[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for target in sorted(graph[node]):
            if target not in indexes:
                strongconnect(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif target in on_stack:
                lowlinks[node] = min(lowlinks[node], indexes[target])

        if lowlinks[node] == indexes[node]:
            component: list[str] = []
            while True:
                target = stack.pop()
                on_stack.remove(target)
                component.append(target)
                if target == node:
                    break
            if len(component) > 1:
                components.append(tuple(sorted(component)))

    for module in sorted(graph):
        if module not in indexes:
            strongconnect(module)
    return sorted(components)


def ring_for_module(module: str) -> str:
    root = module.split(".", 1)[0]
    if module == "main" or root == "api":
        return "api_composition"
    if root == "application":
        return "application"
    if root == "pipeline":
        return "pipeline"
    if root == "providers":
        return "providers"
    if root == "voiceprints":
        return "domain"
    if root == "infra":
        return "infra"
    if root == "config":
        return "configuration"
    if root == "postprocess":
        return "postprocess"
    if root == "nltk":
        return "vendor_shim"
    return "other"


def layer_edges(
    graph: dict[str, set[str]],
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    layer_graph: dict[str, set[str]] = {ring: set() for ring in CORE_RINGS}
    for source, targets in graph.items():
        source_ring = ring_for_module(source)
        for target in targets:
            target_ring = ring_for_module(target)
            if source_ring == target_ring:
                continue
            if source_ring in CORE_RINGS and target_ring in CORE_RINGS:
                grouped[(source_ring, target_ring)].append(
                    {"source": source, "target": target}
                )
                layer_graph[source_ring].add(target_ring)

    edges = [
        {
            "source": source,
            "target": target,
            "imports": sorted(imports, key=lambda item: (item["source"], item["target"])),
        }
        for (source, target), imports in sorted(grouped.items())
    ]
    return edges, layer_graph


def _fastapi_import_labels(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [
            alias.name
            for alias in node.names
            if alias.name == "fastapi" or alias.name.startswith("fastapi.")
        ]
    if isinstance(node, ast.ImportFrom) and node.module:
        if node.module == "fastapi" or node.module.startswith("fastapi."):
            return [f"from {node.module}"]
    return []


def _http_exception_reference_lines(tree: ast.AST) -> list[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == "HTTPException":
            lines.add(node.lineno)
        elif isinstance(node, ast.Attribute) and node.attr == "HTTPException":
            lines.add(node.lineno)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.name == "HTTPException" or alias.asname == "HTTPException":
                    lines.add(node.lineno)
    return sorted(lines)


def forbidden_dependencies(root: Path, graph: dict[str, set[str]]) -> list[dict[str, Any]]:
    module_paths = app_modules(root)
    findings: list[dict[str, Any]] = []

    for module, path in module_paths.items():
        ring = ring_for_module(module)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if ring != "api_composition":
            fastapi_imports: list[str] = []
            for node in ast.walk(tree):
                fastapi_imports.extend(_fastapi_import_labels(node))
            http_exception_lines = _http_exception_reference_lines(tree)
            if fastapi_imports or http_exception_lines:
                findings.append(
                    {
                        "rule": "fastapi_types_stay_in_api_ring",
                        "module": module,
                        "path": str(path.relative_to(root)),
                        "fastapi_imports": fastapi_imports,
                        "http_exception_lines": http_exception_lines,
                    }
                )

    for source, targets in graph.items():
        source_ring = ring_for_module(source)
        for target in sorted(targets):
            if source_ring != "api_composition" and (
                target == "main" or target == "api" or target.startswith("api.")
            ):
                findings.append(
                    {
                        "rule": "non_api_rings_do_not_import_api",
                        "module": source,
                        "target": target,
                    }
                )
            if source_ring == "providers" and (
                target == "application"
                or target.startswith("application.")
                or target == "pipeline.registry"
                or target.startswith("pipeline.registry.")
                or target == "pipeline.stages"
                or target.startswith("pipeline.stages.")
            ):
                findings.append(
                    {
                        "rule": "providers_do_not_import_orchestration_or_stage_registry",
                        "module": source,
                        "target": target,
                    }
                )
    return sorted(findings, key=lambda item: (item["rule"], item.get("module", "")))


def build_report(root: Path) -> dict[str, Any]:
    graph = internal_import_graph(root)
    edges, layer_graph = layer_edges(graph)
    return {
        "module_count": len(graph),
        "internal_edge_count": sum(len(targets) for targets in graph.values()),
        "module_sccs": [list(component) for component in strongly_connected_components(graph)],
        "layer_edges": edges,
        "layer_sccs": [
            list(component) for component in strongly_connected_components(layer_graph)
        ],
        "forbidden_dependencies": forbidden_dependencies(root, graph),
    }


def _print_text(report: dict[str, Any]) -> None:
    print(f"module_count: {report['module_count']}")
    print(f"internal_edge_count: {report['internal_edge_count']}")
    print(f"module_sccs: {report['module_sccs']}")
    print("layer_edges:")
    for edge in report["layer_edges"]:
        print(
            f"- {edge['source']} -> {edge['target']} "
            f"({len(edge['imports'])} imports)"
        )
    print(f"layer_sccs: {report['layer_sccs']}")
    print(f"forbidden_dependencies: {report['forbidden_dependencies']}")


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
        help="Exit non-zero when architecture violations are present.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    report = build_report(root)
    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        _print_text(report)

    if args.check and (
        report["module_sccs"]
        or report["layer_sccs"]
        or report["forbidden_dependencies"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
