"""Source-level architecture gates for import direction constraints."""

from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
NON_API_RING_ROOTS = ("application", "pipeline", "providers", "infra")


def _module_name(path: Path) -> tuple[str, bool]:
    relative = path.relative_to(APP_ROOT).with_suffix("")
    parts = relative.parts
    if parts[-1] == "__init__":
        return ".".join(parts[:-1]), True
    return ".".join(parts), False


def _app_modules() -> dict[str, Path]:
    modules: dict[str, Path] = {}
    for path in sorted(APP_ROOT.rglob("*.py")):
        module_name, _ = _module_name(path)
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


def _is_type_checking_guard(node: ast.expr) -> bool:
    if isinstance(node, ast.Name):
        return node.id == "TYPE_CHECKING"
    if isinstance(node, ast.Attribute):
        return node.attr == "TYPE_CHECKING"
    return False


def _static_internal_import_graph() -> dict[str, set[str]]:
    """Return AST-visible app import edges; dynamic import strings are excluded."""

    module_paths = _app_modules()
    modules = set(module_paths)
    graph: dict[str, set[str]] = {module: set() for module in modules}
    for module, path in module_paths.items():
        _, is_package = _module_name(path)
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


def _strongly_connected_components(graph: dict[str, set[str]]) -> list[tuple[str, ...]]:
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


def _function_call_names(path: Path, function_name: str) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            calls: set[str] = set()
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                if isinstance(child.func, ast.Name):
                    calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.add(child.func.attr)
            return calls
    raise AssertionError(f"{function_name} not found in {path}")


def _non_api_ring_python_files() -> list[Path]:
    paths: list[Path] = []
    for root_name in NON_API_RING_ROOTS:
        root = APP_ROOT / root_name
        if root.exists():
            paths.extend(sorted(root.rglob("*.py")))
    return paths


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


def test_app_internal_static_python_import_graph_has_no_scc():
    graph = _static_internal_import_graph()
    components = _strongly_connected_components(graph)

    assert components == []


def test_non_api_rings_do_not_static_import_fastapi_or_reference_http_exception():
    """Guard source-level API boundary imports; dynamic runtime behavior is separate."""

    offenders: dict[str, dict[str, list[str] | list[int]]] = {}

    for path in _non_api_ring_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        fastapi_imports: list[str] = []
        for node in ast.walk(tree):
            fastapi_imports.extend(_fastapi_import_labels(node))

        http_exception_lines = _http_exception_reference_lines(tree)
        if fastapi_imports or http_exception_lines:
            offenders[str(path.relative_to(REPO_ROOT))] = {
                "fastapi_imports": fastapi_imports,
                "http_exception_lines": http_exception_lines,
            }

    assert offenders == {}


def test_pipeline_contracts_static_imports_stay_on_contracts_or_low_level_pipeline_modules():
    graph = _static_internal_import_graph()
    allowed_exact = {"pipeline.errors", "pipeline.step_keys"}

    def disallowed_targets(targets: set[str]) -> list[str]:
        return sorted(
            target
            for target in targets
            if not (
                target == "pipeline.contracts"
                or target.startswith("pipeline.contracts.")
                or target in allowed_exact
            )
        )

    offenders = {
        module: invalid_targets
        for module, targets in graph.items()
        for invalid_targets in (disallowed_targets(targets),)
        if module.startswith("pipeline.contracts")
        and invalid_targets
    }

    assert offenders == {}


def test_pipeline_registry_static_imports_stay_lazy_across_pipeline_boundaries():
    graph = _static_internal_import_graph()
    forbidden_prefixes = (
        "pipeline.contracts",
        "pipeline.stages",
        "providers",
    )
    offenders = sorted(
        target
        for target in graph["pipeline.registry"]
        if any(
            target == prefix or target.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
        )
    )

    assert offenders == []


def test_provider_selector_normalizers_delegate_to_shared_token_normalizer():
    module_paths = _app_modules()
    checked_modules = (
        "pipeline.contracts.requests",
        "providers.capabilities",
    )
    manual_calls: dict[str, list[str]] = {}
    missing_delegate: list[str] = []

    for module in checked_modules:
        calls = _function_call_names(module_paths[module], "_normalize_provider_name")
        if "normalize_token" not in calls:
            missing_delegate.append(module)
        duplicated_steps = sorted({"strip", "lower", "replace"} & calls)
        if duplicated_steps:
            manual_calls[module] = duplicated_steps

    assert missing_delegate == []
    assert manual_calls == {}


def test_pipeline_lookup_errors_keep_legacy_public_imports():
    from pipeline.contracts import (
        ProviderNotFoundError as ContractsProviderNotFoundError,
    )
    from pipeline.contracts import StageNotFoundError as ContractsStageNotFoundError
    from pipeline.registry import ProviderNotFoundError as RegistryProviderNotFoundError
    from pipeline.registry import StageNotFoundError as RegistryStageNotFoundError

    assert ContractsProviderNotFoundError is RegistryProviderNotFoundError
    assert ContractsStageNotFoundError is RegistryStageNotFoundError
