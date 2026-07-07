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

REGISTRY_RUNTIME_IMPORTS = {
    "pipeline.registry": {
        "_DEFAULT_STAGE_IMPORTS": "registry_stage",
        "_DEFAULT_PROVIDER_IMPORTS": "registry_provider",
    },
}

STATUS_CONTRACT_HELPERS = frozenset(
    {
        "IN_PROGRESS_JOB_STATUSES",
        "JOB_STATUS_COMPLETED",
        "JOB_STATUS_CONVERTING",
        "JOB_STATUS_DENOISING",
        "JOB_STATUS_FAILED",
        "JOB_STATUS_IDENTIFYING",
        "JOB_STATUS_QUEUED",
        "JOB_STATUS_TRANSCRIBING",
        "KNOWN_JOB_STATUSES",
        "TERMINAL_JOB_STATUSES",
        "build_status_payload",
        "normalize_job_status",
        "normalize_status_payload",
    }
)

PIPELINE_METADATA_CONTRACT_MODULE = "pipeline.contracts.metadata"


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
        collector.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        graph[module].update(collector.targets)
    return graph


def _import_module_bindings(tree: ast.AST) -> tuple[set[str], set[str]]:
    direct_names: set[str] = set()
    module_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "importlib":
            for alias in node.names:
                if alias.name == "import_module":
                    direct_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "importlib":
                    module_names.add(alias.asname or alias.name)
    return direct_names, module_names


def _is_import_module_call(
    call: ast.Call,
    *,
    direct_names: set[str],
    module_names: set[str],
) -> bool:
    if isinstance(call.func, ast.Name):
        return call.func.id in direct_names
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "import_module":
        return False
    return isinstance(call.func.value, ast.Name) and call.func.value.id in module_names


def _iter_string_constants(node: ast.AST) -> list[ast.Constant]:
    strings: list[ast.Constant] = []
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        strings.append(node)
    elif isinstance(node, ast.Dict):
        for value in node.values:
            if value is not None:
                strings.extend(_iter_string_constants(value))
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        for value in node.elts:
            strings.extend(_iter_string_constants(value))
    return strings


def _runtime_module_name(import_value: str) -> str:
    module_name, _, _ = import_value.partition(":")
    return module_name.strip()


def _add_runtime_edge(
    edges_by_key: dict[tuple[str, str, str, str], dict[str, Any]],
    *,
    root: Path,
    source: str,
    target: str,
    kind: str,
    import_value: str,
    path: Path,
    lineno: int,
) -> None:
    if target == source:
        return
    key = (source, target, kind, import_value)
    edge = edges_by_key.setdefault(
        key,
        {
            "source": source,
            "target": target,
            "kind": kind,
            "import": import_value,
            "locations": [],
        },
    )
    edge["locations"].append({"path": str(path.relative_to(root)), "line": lineno})


def _registry_runtime_edges(
    *,
    root: Path,
    module: str,
    path: Path,
    tree: ast.AST,
    modules: set[str],
    edges_by_key: dict[tuple[str, str, str, str], dict[str, Any]],
) -> None:
    registry_imports = REGISTRY_RUNTIME_IMPORTS.get(module)
    if not registry_imports:
        return

    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        target_names = {
            target.id for target in node.targets if isinstance(target, ast.Name)
        }
        for assignment_name in sorted(target_names & registry_imports.keys()):
            kind = registry_imports[assignment_name]
            for string_node in _iter_string_constants(node.value):
                import_value = _runtime_module_name(string_node.value)
                target = _internal_module_for(import_value, modules)
                if target is not None:
                    _add_runtime_edge(
                        edges_by_key,
                        root=root,
                        source=module,
                        target=target,
                        kind=kind,
                        import_value=import_value,
                        path=path,
                        lineno=string_node.lineno,
                    )


def _literal_import_module_edges(
    *,
    root: Path,
    module: str,
    path: Path,
    tree: ast.AST,
    modules: set[str],
    edges_by_key: dict[tuple[str, str, str, str], dict[str, Any]],
) -> None:
    direct_names, module_names = _import_module_bindings(tree)
    if not direct_names and not module_names:
        return

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if not _is_import_module_call(
            node,
            direct_names=direct_names,
            module_names=module_names,
        ):
            continue
        import_arg = node.args[0]
        if not isinstance(import_arg, ast.Constant) or not isinstance(
            import_arg.value,
            str,
        ):
            continue
        import_value = _runtime_module_name(import_arg.value)
        target = _internal_module_for(import_value, modules)
        if target is not None:
            _add_runtime_edge(
                edges_by_key,
                root=root,
                source=module,
                target=target,
                kind="literal_import_module",
                import_value=import_value,
                path=path,
                lineno=import_arg.lineno,
            )


def runtime_dynamic_import_edges(root: Path) -> list[dict[str, Any]]:
    module_paths = app_modules(root)
    modules = set(module_paths)
    edges_by_key: dict[tuple[str, str, str, str], dict[str, Any]] = {}

    for module, path in module_paths.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        _registry_runtime_edges(
            root=root,
            module=module,
            path=path,
            tree=tree,
            modules=modules,
            edges_by_key=edges_by_key,
        )
        _literal_import_module_edges(
            root=root,
            module=module,
            path=path,
            tree=tree,
            modules=modules,
            edges_by_key=edges_by_key,
        )

    edges = sorted(
        edges_by_key.values(),
        key=lambda item: (
            item["source"],
            item["target"],
            item["kind"],
            item["import"],
        ),
    )
    for edge in edges:
        edge["locations"] = sorted(
            edge["locations"],
            key=lambda item: (item["path"], item["line"]),
        )
    return edges


def runtime_dynamic_import_graph(root: Path) -> dict[str, set[str]]:
    graph: dict[str, set[str]] = {module: set() for module in app_modules(root)}
    for edge in runtime_dynamic_import_edges(root):
        graph[edge["source"]].add(edge["target"])
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
            "imports": sorted(
                imports, key=lambda item: (item["source"], item["target"])
            ),
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


def _pipeline_status_import_locations(tree: ast.AST) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pipeline.contracts.status":
                    locations.append(
                        {
                            "line": node.lineno,
                            "import": alias.name,
                            "symbol": None,
                        }
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module == "pipeline.contracts.status":
                imported = tuple(alias.name for alias in node.names)
                locations.append(
                    {
                        "line": node.lineno,
                        "import": "from pipeline.contracts.status",
                        "symbol": "*" if "*" in imported else ", ".join(imported),
                    }
                )
            elif node.module == "pipeline.contracts":
                for alias in node.names:
                    if (
                        alias.name == "*"
                        or alias.name == "status"
                        or alias.name in STATUS_CONTRACT_HELPERS
                    ):
                        locations.append(
                            {
                                "line": node.lineno,
                                "import": "from pipeline.contracts",
                                "symbol": alias.name,
                            }
                        )
    return locations


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base is None:
            return None
        return f"{base}.{node.attr}"
    return None


def _is_private_job_persistence_symbol(name: str) -> bool:
    return name.startswith("_")


def _application_job_boundary_locations(tree: ast.AST) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    runtime_module_aliases: set[str] = set()
    persistence_module_aliases: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_name = alias.asname or alias.name
                if alias.name == "infra.job_runtime":
                    runtime_module_aliases.add(imported_name)
                elif alias.name == "infra.job_persistence":
                    persistence_module_aliases.add(imported_name)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "infra.job_runtime":
                for alias in node.names:
                    if alias.name in {"*", "jobs"}:
                        locations.append(
                            {
                                "line": node.lineno,
                                "import": "from infra.job_runtime",
                                "symbol": alias.name,
                            }
                        )
            elif node.module == "infra.job_persistence":
                for alias in node.names:
                    if alias.name == "*" or _is_private_job_persistence_symbol(
                        alias.name
                    ):
                        locations.append(
                            {
                                "line": node.lineno,
                                "import": "from infra.job_persistence",
                                "symbol": alias.name,
                            }
                        )
            elif node.module == "infra":
                for alias in node.names:
                    imported_name = alias.asname or alias.name
                    if alias.name == "job_runtime":
                        runtime_module_aliases.add(imported_name)
                    elif alias.name == "job_persistence":
                        persistence_module_aliases.add(imported_name)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        base_name = _dotted_name(node.value)
        if node.attr == "jobs" and base_name in runtime_module_aliases:
            locations.append(
                {
                    "line": node.lineno,
                    "import": f"{base_name}.jobs",
                    "symbol": "jobs",
                }
            )
        elif (
            _is_private_job_persistence_symbol(node.attr)
            and base_name in persistence_module_aliases
        ):
            locations.append(
                {
                    "line": node.lineno,
                    "import": f"{base_name}.{node.attr}",
                    "symbol": node.attr,
                }
            )

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value in {"result.json", "status.json"}
        ):
            locations.append(
                {
                    "line": node.lineno,
                    "import": "transcription record filesystem literal",
                    "symbol": node.value,
                }
            )

    return sorted(
        locations,
        key=lambda item: (item["line"], item["import"], item["symbol"] or ""),
    )


def _pipeline_metadata_allowed_top_level_keys(root: Path) -> frozenset[str]:
    metadata_contract = root / "app" / "pipeline" / "contracts" / "metadata.py"
    if not metadata_contract.exists():
        return frozenset()

    tree = ast.parse(metadata_contract.read_text(encoding="utf-8"))
    assignments: dict[str, tuple[str, ...]] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        if target.id in {
            "PIPELINE_METADATA_CONTROL_KEYS",
            "PIPELINE_METADATA_STAGE_KEYS",
            "PIPELINE_METADATA_TOP_LEVEL_KEYS",
        }:
            assignments[target.id] = _literal_string_tuple(
                node.value,
                assignments,
            )
    if assignments.get("PIPELINE_METADATA_TOP_LEVEL_KEYS"):
        return frozenset(assignments["PIPELINE_METADATA_TOP_LEVEL_KEYS"])
    return frozenset(
        (
            *assignments.get("PIPELINE_METADATA_CONTROL_KEYS", ()),
            *assignments.get("PIPELINE_METADATA_STAGE_KEYS", ()),
        )
    )


def _literal_string_tuple(
    node: ast.AST,
    assignments: dict[str, tuple[str, ...]],
) -> tuple[str, ...]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return (node.value,)
    if isinstance(node, ast.Name):
        return assignments.get(node.id, ())
    if isinstance(node, ast.Starred):
        return _literal_string_tuple(node.value, assignments)
    if isinstance(node, (ast.Tuple, ast.List)):
        values: list[str] = []
        for element in node.elts:
            values.extend(_literal_string_tuple(element, assignments))
        return tuple(values)
    return ()


def _literal_string_slice(node: ast.slice) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_context_metadata(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "metadata"
        and isinstance(node.value, ast.Name)
        and node.value.id == "context"
    )


def _context_metadata_subscript_key(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Subscript) or not _is_context_metadata(node.value):
        return None
    return _literal_string_slice(node.slice)


def _context_metadata_update_key(node: ast.AST) -> str | None:
    if (
        not isinstance(node, ast.Call)
        or not isinstance(node.func, ast.Attribute)
        or node.func.attr != "update"
        or not isinstance(node.func.value, ast.Subscript)
        or not _is_context_metadata(node.func.value.value)
    ):
        return None
    return _literal_string_slice(node.func.value.slice) or "<dynamic>"


def _pipeline_context_metadata_key_locations(
    *,
    tree: ast.AST,
    allowed_keys: frozenset[str],
) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Subscript):
            key = _context_metadata_subscript_key(node)
            if key is not None and key not in allowed_keys:
                locations.append(
                    {
                        "line": node.lineno,
                        "key": key,
                        "access": "subscript",
                    }
                )
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in {"get", "pop", "setdefault"}
            and _is_context_metadata(node.func.value)
            and node.args
        ):
            key_node = node.args[0]
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                key = key_node.value
                if key not in allowed_keys:
                    locations.append(
                        {
                            "line": node.lineno,
                            "key": key,
                            "access": node.func.attr,
                        }
                    )

    return sorted(
        locations,
        key=lambda item: (item["line"], item["key"], item["access"]),
    )


def _pipeline_context_metadata_update_locations(
    *,
    tree: ast.AST,
) -> list[dict[str, Any]]:
    locations: list[dict[str, Any]] = []
    for node in ast.walk(tree):
        key = _context_metadata_update_key(node)
        if key is not None:
            locations.append(
                {
                    "line": node.lineno,
                    "key": key,
                }
            )
    return sorted(locations, key=lambda item: (item["line"], item["key"]))


def _assignment_value(tree: ast.AST, name: str) -> ast.AST | None:
    body = tree.body if isinstance(tree, ast.Module) else ()
    for node in body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == name
        ):
            return node.value
    return None


def _string_constant(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _keyword_string(call: ast.Call, name: str) -> str | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return _string_constant(keyword.value)
    return None


def _registry_stage_slots(root: Path) -> dict[str, dict[str, Any]]:
    registry_path = root / "app" / "pipeline" / "registry.py"
    if not registry_path.exists():
        return {}

    tree = ast.parse(registry_path.read_text(encoding="utf-8"))
    assignment = _assignment_value(tree, "_DEFAULT_STAGE_IMPORTS")
    if not isinstance(assignment, ast.Dict):
        return {}

    slots: dict[str, dict[str, Any]] = {}
    for key_node in assignment.keys:
        stage = _string_constant(key_node) if key_node is not None else None
        if stage is not None:
            slots[stage] = {
                "path": str(registry_path.relative_to(root)),
                "line": key_node.lineno,
            }
    return slots


def _registry_provider_surface(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    registry_path = root / "app" / "pipeline" / "registry.py"
    if not registry_path.exists():
        return {}

    tree = ast.parse(registry_path.read_text(encoding="utf-8"))
    assignment = _assignment_value(tree, "_DEFAULT_PROVIDER_IMPORTS")
    if not isinstance(assignment, ast.Dict):
        return {}

    providers: dict[tuple[str, str], dict[str, Any]] = {}
    for stage_node, provider_map in zip(assignment.keys, assignment.values):
        stage = _string_constant(stage_node) if stage_node is not None else None
        if stage is None or not isinstance(provider_map, ast.Dict):
            continue
        for provider_node in provider_map.keys:
            provider = (
                _string_constant(provider_node) if provider_node is not None else None
            )
            if provider is not None:
                providers[(stage, provider)] = {
                    "path": str(registry_path.relative_to(root)),
                    "line": provider_node.lineno,
                }
    return providers


def _provider_capability_surface(root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    capabilities_path = root / "app" / "providers" / "capabilities.py"
    if not capabilities_path.exists():
        return {}

    tree = ast.parse(capabilities_path.read_text(encoding="utf-8"))
    assignment = _assignment_value(tree, "_DEFAULT_CAPABILITIES")
    if not isinstance(assignment, ast.Dict):
        return {}

    capabilities: dict[tuple[str, str], dict[str, Any]] = {}
    for key_node, value_node in zip(assignment.keys, assignment.values):
        if (
            not isinstance(key_node, ast.Tuple)
            or len(key_node.elts) != 2
            or not isinstance(value_node, ast.Call)
        ):
            continue
        stage_key = _string_constant(key_node.elts[0])
        provider_key = _string_constant(key_node.elts[1])
        if stage_key is None or provider_key is None:
            continue
        capabilities[(stage_key, provider_key)] = {
            "path": str(capabilities_path.relative_to(root)),
            "line": key_node.lineno,
            "declared_stage": _keyword_string(value_node, "stage") or stage_key,
            "declared_name": _keyword_string(value_node, "name") or provider_key,
            "capability": _keyword_string(value_node, "capability"),
        }
    return capabilities


def provider_capability_contract_findings(root: Path) -> list[dict[str, Any]]:
    """Ensure registry-selectable providers and static capability metadata match."""

    stage_slots = _registry_stage_slots(root)
    registry_providers = _registry_provider_surface(root)
    capabilities = _provider_capability_surface(root)
    findings: list[dict[str, Any]] = []

    for key, registry_location in sorted(registry_providers.items()):
        stage, provider = key
        capability = capabilities.get(key)
        if capability is None:
            findings.append(
                {
                    "rule": "provider_registry_has_static_capability_record",
                    "stage": stage,
                    "provider": provider,
                    "path": registry_location["path"],
                    "line": registry_location["line"],
                }
            )
            continue
        if (
            capability["declared_stage"] != stage
            or capability["declared_name"] != provider
        ):
            findings.append(
                {
                    "rule": "provider_capability_matches_registry_key",
                    "stage": stage,
                    "provider": provider,
                    "capability_stage": capability["declared_stage"],
                    "capability_provider": capability["declared_name"],
                    "path": capability["path"],
                    "line": capability["line"],
                }
            )

    for key, capability in sorted(capabilities.items()):
        if key in registry_providers:
            continue
        declared_stage = capability["declared_stage"]
        if capability["capability"] is not None and declared_stage in stage_slots:
            continue
        findings.append(
            {
                "rule": "provider_capability_has_registry_provider_or_stage_owner",
                "stage": key[0],
                "provider": key[1],
                "capability_stage": declared_stage,
                "capability": capability["capability"],
                "path": capability["path"],
                "line": capability["line"],
            }
        )
    return findings


def forbidden_dependencies(
    root: Path, graph: dict[str, set[str]]
) -> list[dict[str, Any]]:
    module_paths = app_modules(root)
    findings: list[dict[str, Any]] = []
    pipeline_metadata_allowed_keys = _pipeline_metadata_allowed_top_level_keys(root)

    for module, path in module_paths.items():
        ring = ring_for_module(module)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        metadata_key_locations = _pipeline_context_metadata_key_locations(
            tree=tree,
            allowed_keys=pipeline_metadata_allowed_keys,
        )
        if metadata_key_locations:
            findings.append(
                {
                    "rule": "pipeline_context_metadata_top_level_key_contract",
                    "module": module,
                    "path": str(path.relative_to(root)),
                    "locations": metadata_key_locations,
                }
            )
        if module != PIPELINE_METADATA_CONTRACT_MODULE:
            metadata_update_locations = _pipeline_context_metadata_update_locations(
                tree=tree,
            )
            if metadata_update_locations:
                findings.append(
                    {
                        "rule": "pipeline_context_metadata_no_unbounded_update",
                        "module": module,
                        "path": str(path.relative_to(root)),
                        "locations": metadata_update_locations,
                    }
                )
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
        if ring in {"application", "infra"}:
            status_imports = _pipeline_status_import_locations(tree)
            if status_imports:
                findings.append(
                    {
                        "rule": "application_and_infra_use_infra_job_status_owner",
                        "module": module,
                        "path": str(path.relative_to(root)),
                        "locations": status_imports,
                    }
                )
        if ring == "application":
            job_boundary_imports = _application_job_boundary_locations(tree)
            if job_boundary_imports:
                findings.append(
                    {
                        "rule": "application_uses_public_infra_job_boundary",
                        "module": module,
                        "path": str(path.relative_to(root)),
                        "locations": job_boundary_imports,
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
    findings.extend(provider_capability_contract_findings(root))
    return sorted(findings, key=lambda item: (item["rule"], item.get("module", "")))


def forbidden_dynamic_dependencies(
    dynamic_edges: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for edge in dynamic_edges:
        source = edge["source"]
        target = edge["target"]
        source_ring = ring_for_module(source)
        target_ring = ring_for_module(target)
        provider_imports_orchestration = source_ring == "providers" and (
            target_ring == "application"
            or target == "pipeline.registry"
            or target.startswith("pipeline.registry.")
            or target == "pipeline.stages"
            or target.startswith("pipeline.stages.")
        )
        if source_ring != "api_composition" and (
            target == "main" or target == "api" or target.startswith("api.")
        ):
            findings.append(
                {
                    "rule": "non_api_rings_do_not_runtime_import_api",
                    "module": source,
                    "target": target,
                    "kind": edge["kind"],
                    "import": edge["import"],
                    "locations": edge["locations"],
                }
            )
        if provider_imports_orchestration:
            findings.append(
                {
                    "rule": "providers_do_not_runtime_import_orchestration_or_stage_registry",
                    "module": source,
                    "target": target,
                    "kind": edge["kind"],
                    "import": edge["import"],
                    "locations": edge["locations"],
                }
            )
        elif source_ring not in {"api_composition", "application"} and (
            target_ring == "application"
        ):
            findings.append(
                {
                    "rule": "non_application_rings_do_not_runtime_import_application",
                    "module": source,
                    "target": target,
                    "kind": edge["kind"],
                    "import": edge["import"],
                    "locations": edge["locations"],
                }
            )
    return sorted(
        findings,
        key=lambda item: (item["rule"], item["module"], item["target"]),
    )


def build_report(root: Path) -> dict[str, Any]:
    static_graph = internal_import_graph(root)
    static_edges, static_layer_graph = layer_edges(static_graph)
    dynamic_edges = runtime_dynamic_import_edges(root)
    dynamic_graph: dict[str, set[str]] = {module: set() for module in static_graph}
    for edge in dynamic_edges:
        dynamic_graph[edge["source"]].add(edge["target"])
    dynamic_layer_edges, _ = layer_edges(dynamic_graph)
    return {
        "static_import_graph": {
            "module_count": len(static_graph),
            "internal_edge_count": sum(
                len(targets) for targets in static_graph.values()
            ),
            "module_sccs": [
                list(component)
                for component in strongly_connected_components(static_graph)
            ],
            "layer_edges": static_edges,
            "layer_sccs": [
                list(component)
                for component in strongly_connected_components(static_layer_graph)
            ],
        },
        "static_forbidden_dependencies": forbidden_dependencies(root, static_graph),
        "runtime_dynamic_import_graph": {
            "edge_count": sum(len(targets) for targets in dynamic_graph.values()),
            "edges": dynamic_edges,
            "module_sccs": [
                list(component)
                for component in strongly_connected_components(dynamic_graph)
            ],
            "layer_edges": dynamic_layer_edges,
        },
        "runtime_dynamic_forbidden_dependencies": forbidden_dynamic_dependencies(
            dynamic_edges
        ),
    }


def _print_text(report: dict[str, Any]) -> None:
    static_graph = report["static_import_graph"]
    print("static_import_graph:")
    print(f"  module_count: {static_graph['module_count']}")
    print(f"  internal_edge_count: {static_graph['internal_edge_count']}")
    print(f"  module_sccs: {static_graph['module_sccs']}")
    print("  layer_edges:")
    for edge in static_graph["layer_edges"]:
        print(
            f"  - {edge['source']} -> {edge['target']} ({len(edge['imports'])} imports)"
        )
    print(f"  layer_sccs: {static_graph['layer_sccs']}")
    print(f"static_forbidden_dependencies: {report['static_forbidden_dependencies']}")

    dynamic_graph = report["runtime_dynamic_import_graph"]
    print("runtime_dynamic_import_graph:")
    print(f"  edge_count: {dynamic_graph['edge_count']}")
    print("  edges:")
    for edge in dynamic_graph["edges"]:
        locations = ", ".join(
            f"{location['path']}:{location['line']}" for location in edge["locations"]
        )
        print(
            f"  - {edge['source']} -> {edge['target']} "
            f"[{edge['kind']}] import={edge['import']!r} ({locations})"
        )
    print(f"  module_sccs: {dynamic_graph['module_sccs']}")
    print("  layer_edges:")
    for edge in dynamic_graph["layer_edges"]:
        print(
            f"  - {edge['source']} -> {edge['target']} "
            f"({len(edge['imports'])} runtime imports)"
        )
    print(
        "runtime_dynamic_forbidden_dependencies: "
        f"{report['runtime_dynamic_forbidden_dependencies']}"
    )


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

    static_graph = report["static_import_graph"]
    dynamic_graph = report["runtime_dynamic_import_graph"]
    if args.check and (
        static_graph["module_sccs"]
        or static_graph["layer_sccs"]
        or report["static_forbidden_dependencies"]
        or dynamic_graph["module_sccs"]
        or report["runtime_dynamic_forbidden_dependencies"]
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
