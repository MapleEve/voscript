"""Source-level architecture gates for import direction constraints."""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
APP_ROOT = REPO_ROOT / "app"
NON_API_RING_ROOTS = (
    "application",
    "pipeline",
    "providers",
    "infra",
    "voiceprints",
    "postprocess",
)
ARCHITECTURE_GATE = REPO_ROOT / "voscript-api" / "scripts" / "architecture_gate.py"


def _load_architecture_gate():
    spec = importlib.util.spec_from_file_location(
        "voscript_architecture_gate",
        ARCHITECTURE_GATE,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
        collector.visit(ast.parse(path.read_text(encoding="utf-8"), filename=str(path)))
        graph[module].update(collector.targets)
    return graph


def _runtime_dynamic_edge_keys(report: dict) -> set[tuple[str, str, str]]:
    return {
        (edge["source"], edge["target"], edge["kind"])
        for edge in report["runtime_dynamic_import_graph"]["edges"]
    }


def _runtime_dynamic_forbidden_keys(report: dict) -> set[tuple[str, str, str]]:
    return {
        (finding["rule"], finding["module"], finding["target"])
        for finding in report["runtime_dynamic_forbidden_dependencies"]
    }


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


def _pipeline_status_import_locations(tree: ast.AST) -> list[dict[str, object]]:
    gate = _load_architecture_gate()
    return gate._pipeline_status_import_locations(tree)


def test_app_internal_static_python_import_graph_has_no_scc():
    graph = _static_internal_import_graph()
    components = _strongly_connected_components(graph)

    assert components == []


def test_architecture_gate_report_exposes_cycle_evidence():
    gate = _load_architecture_gate()
    report = gate.build_report(REPO_ROOT)
    static_graph = report["static_import_graph"]
    dynamic_graph = report["runtime_dynamic_import_graph"]

    assert static_graph["module_count"] == len(_app_modules())
    assert static_graph["internal_edge_count"] == sum(
        len(targets) for targets in _static_internal_import_graph().values()
    )
    assert set(report) == {
        "runtime_dynamic_forbidden_dependencies",
        "runtime_dynamic_import_graph",
        "static_forbidden_dependencies",
        "static_import_graph",
    }
    assert set(static_graph) == {
        "internal_edge_count",
        "layer_edges",
        "layer_sccs",
        "module_count",
        "module_sccs",
    }
    assert set(dynamic_graph) == {
        "edge_count",
        "edges",
        "layer_edges",
        "module_sccs",
    }
    assert static_graph["module_sccs"] == []
    assert static_graph["layer_sccs"] == []
    assert report["static_forbidden_dependencies"] == []
    assert dynamic_graph["module_sccs"] == []
    assert report["runtime_dynamic_forbidden_dependencies"] == []


def test_architecture_gate_reports_runtime_dynamic_registry_and_literal_edges():
    gate = _load_architecture_gate()
    report = gate.build_report(REPO_ROOT)
    edge_keys = _runtime_dynamic_edge_keys(report)

    assert (
        "pipeline.registry",
        "pipeline.stages.asr",
        "registry_stage",
    ) in edge_keys
    assert (
        "pipeline.registry",
        "providers.asr.default",
        "registry_provider",
    ) in edge_keys
    assert (
        "pipeline.runner",
        "infra.audio",
        "literal_import_module",
    ) in edge_keys
    assert (
        "pipeline.runner",
        "providers.capabilities",
        "literal_import_module",
    ) in edge_keys
    assert (
        "pipeline.orchestrator",
        "infra.huggingface_models",
        "literal_import_module",
    ) in edge_keys
    assert (
        "pipeline.orchestrator",
        "infra.cuda_devices",
        "literal_import_module",
    ) in edge_keys
    assert (
        "pipeline.orchestrator",
        "providers.asr",
        "literal_import_module",
    ) in edge_keys
    assert (
        "pipeline.orchestrator",
        "providers.diarization",
        "literal_import_module",
    ) in edge_keys
    assert (
        "pipeline.orchestrator",
        "providers.embedding",
        "literal_import_module",
    ) in edge_keys
    assert (
        "providers._registry",
        "pipeline.registry",
        "literal_import_module",
    ) not in edge_keys


def _write_module(root: Path, relative_path: str, source: str) -> None:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _write_metadata_contract(root: Path, keys: tuple[str, ...]) -> None:
    _write_module(
        root,
        "app/pipeline/contracts/metadata.py",
        f"PIPELINE_METADATA_TOP_LEVEL_KEYS = {keys!r}\n",
    )


def test_architecture_gate_flags_unknown_pipeline_context_metadata_keys(tmp_path):
    gate = _load_architecture_gate()
    _write_metadata_contract(tmp_path, ("diarization",))
    _write_module(
        tmp_path,
        "app/pipeline/stages/example.py",
        "def run(context):\n    context.metadata['freeform'] = {'status': 'bad'}\n",
    )

    report = gate.build_report(tmp_path)

    assert report["static_forbidden_dependencies"] == [
        {
            "rule": "pipeline_context_metadata_top_level_key_contract",
            "module": "pipeline.stages.example",
            "path": "app/pipeline/stages/example.py",
            "locations": [
                {
                    "line": 2,
                    "key": "freeform",
                    "access": "subscript",
                }
            ],
        }
    ]


def test_architecture_gate_reads_metadata_contract_without_executing_it(tmp_path):
    gate = _load_architecture_gate()
    _write_module(
        tmp_path,
        "app/pipeline/contracts/metadata.py",
        "PIPELINE_METADATA_CONTROL_KEYS = ('executed_stages',)\n"
        "PIPELINE_METADATA_STAGE_KEYS = ('diarization',)\n"
        "PIPELINE_METADATA_TOP_LEVEL_KEYS = (\n"
        "    *PIPELINE_METADATA_CONTROL_KEYS,\n"
        "    *PIPELINE_METADATA_STAGE_KEYS,\n"
        ")\n"
        "raise RuntimeError('metadata contract must not execute')\n",
    )
    _write_module(
        tmp_path,
        "app/pipeline/stages/example.py",
        "def run(context):\n    context.metadata['diarization'] = {'status': 'ok'}\n",
    )

    report = gate.build_report(tmp_path)

    assert report["static_forbidden_dependencies"] == []


def test_architecture_gate_flags_unbounded_pipeline_context_metadata_update(tmp_path):
    gate = _load_architecture_gate()
    _write_metadata_contract(tmp_path, ("diarization",))
    _write_module(
        tmp_path,
        "app/pipeline/stages/example.py",
        "def run(context, result):\n"
        "    context.metadata['diarization'].update(result.metadata)\n",
    )

    report = gate.build_report(tmp_path)

    assert report["static_forbidden_dependencies"] == [
        {
            "rule": "pipeline_context_metadata_no_unbounded_update",
            "module": "pipeline.stages.example",
            "path": "app/pipeline/stages/example.py",
            "locations": [
                {
                    "line": 2,
                    "key": "diarization",
                }
            ],
        }
    ]


def test_architecture_gate_flags_runtime_dynamic_import_scc(tmp_path):
    gate = _load_architecture_gate()
    _write_module(
        tmp_path,
        "app/pipeline/a.py",
        "from importlib import import_module\n\n\ndef load():\n    return import_module('pipeline.b')\n",
    )
    _write_module(
        tmp_path,
        "app/pipeline/b.py",
        "from importlib import import_module\n\n\ndef load():\n    return import_module('pipeline.a')\n",
    )

    report = gate.build_report(tmp_path)

    assert report["static_import_graph"]["module_sccs"] == []
    assert report["runtime_dynamic_import_graph"]["module_sccs"] == [
        ["pipeline.a", "pipeline.b"]
    ]


def test_architecture_gate_flags_runtime_dynamic_application_boundary(tmp_path):
    gate = _load_architecture_gate()
    _write_module(tmp_path, "app/application/jobs.py", "def run():\n    return None\n")
    _write_module(
        tmp_path,
        "app/providers/default.py",
        "from importlib import import_module\n\n\ndef load():\n    return import_module('application.jobs')\n",
    )

    report = gate.build_report(tmp_path)

    assert report["runtime_dynamic_forbidden_dependencies"] == [
        {
            "rule": "providers_do_not_runtime_import_orchestration_or_stage_registry",
            "module": "providers.default",
            "target": "application.jobs",
            "kind": "literal_import_module",
            "import": "application.jobs",
            "locations": [
                {
                    "path": "app/providers/default.py",
                    "line": 5,
                }
            ],
        }
    ]


def test_architecture_gate_flags_provider_runtime_registry_and_stage_imports(tmp_path):
    gate = _load_architecture_gate()
    _write_module(
        tmp_path, "app/pipeline/registry.py", "def resolve_provider():\n    pass\n"
    )
    _write_module(tmp_path, "app/pipeline/stages/asr.py", "def run():\n    pass\n")
    _write_module(
        tmp_path,
        "app/providers/default.py",
        "from importlib import import_module\n\n\n"
        "def load_registry():\n"
        "    return import_module('pipeline.registry')\n\n\n"
        "def load_stage():\n"
        "    return import_module('pipeline.stages.asr')\n",
    )

    report = gate.build_report(tmp_path)

    assert {
        (
            "providers_do_not_runtime_import_orchestration_or_stage_registry",
            "providers.default",
            "pipeline.registry",
        ),
        (
            "providers_do_not_runtime_import_orchestration_or_stage_registry",
            "providers.default",
            "pipeline.stages.asr",
        ),
    }.issubset(_runtime_dynamic_forbidden_keys(report))


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
        if module.startswith("pipeline.contracts") and invalid_targets
    }

    assert offenders == {}


def test_status_contract_owner_is_infra_not_pipeline_contracts():
    assert not (APP_ROOT / "pipeline" / "contracts" / "status.py").exists()

    from infra import job_status
    from pipeline import contracts

    assert job_status.build_status_payload(
        "queued",
        filename="private/audio.wav",
        updated_at="2026-06-09T00:00:00+00:00",
    ) == {
        "status": "queued",
        "updated_at": "2026-06-09T00:00:00+00:00",
        "error": None,
        "filename": "audio.wav",
    }
    forbidden_exports = {
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
    assert [
        name for name in sorted(forbidden_exports) if hasattr(contracts, name)
    ] == []


def test_application_and_infra_do_not_import_pipeline_status_helpers():
    offenders: dict[str, list[dict[str, object]]] = {}

    for root_name in ("application", "infra"):
        root = APP_ROOT / root_name
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            status_imports = _pipeline_status_import_locations(tree)
            if status_imports:
                offenders[str(path.relative_to(REPO_ROOT))] = status_imports

    assert offenders == {}


def test_architecture_gate_flags_application_or_infra_status_contract_imports(
    tmp_path,
):
    gate = _load_architecture_gate()
    _write_module(
        tmp_path,
        "app/application/records.py",
        "from pipeline.contracts import normalize_status_payload\n",
    )
    _write_module(
        tmp_path,
        "app/infra/jobs.py",
        "from pipeline.contracts.status import build_status_payload\n",
    )
    _write_module(
        tmp_path,
        "app/pipeline/contracts/status.py",
        "def build_status_payload():\n    pass\n",
    )

    report = gate.build_report(tmp_path)

    assert report["static_forbidden_dependencies"] == [
        {
            "rule": "application_and_infra_use_infra_job_status_owner",
            "module": "application.records",
            "path": "app/application/records.py",
            "locations": [
                {
                    "line": 1,
                    "import": "from pipeline.contracts",
                    "symbol": "normalize_status_payload",
                }
            ],
        },
        {
            "rule": "application_and_infra_use_infra_job_status_owner",
            "module": "infra.jobs",
            "path": "app/infra/jobs.py",
            "locations": [
                {
                    "line": 1,
                    "import": "from pipeline.contracts.status",
                    "symbol": "build_status_payload",
                }
            ],
        },
    ]


def test_architecture_gate_flags_application_private_job_boundary_imports(tmp_path):
    gate = _load_architecture_gate()
    _write_module(
        tmp_path,
        "app/application/jobs.py",
        "from infra.job_runtime import jobs\n"
        "from infra.job_persistence import _write_status, write_job_status\n"
        "from infra.job_persistence import *\n\n"
        "def record_files():\n"
        "    return 'status.json', 'result.json'\n",
    )
    _write_module(
        tmp_path,
        "app/application/runtime_module.py",
        "import infra.job_runtime as job_runtime\n\n"
        "def current_jobs():\n"
        "    return job_runtime.jobs\n",
    )
    _write_module(
        tmp_path,
        "app/application/persistence_module.py",
        "import infra.job_persistence as job_persistence\n\n"
        "def atomic_write():\n"
        "    return job_persistence._atomic_write_json\n",
    )
    _write_module(
        tmp_path,
        "app/application/runtime_from_infra.py",
        "from infra import job_runtime\n\n"
        "def current_jobs():\n"
        "    return job_runtime.jobs\n",
    )

    report = gate.build_report(tmp_path)

    assert report["static_forbidden_dependencies"] == [
        {
            "rule": "application_uses_public_infra_job_boundary",
            "module": "application.jobs",
            "path": "app/application/jobs.py",
            "locations": [
                {
                    "line": 1,
                    "import": "from infra.job_runtime",
                    "symbol": "jobs",
                },
                {
                    "line": 2,
                    "import": "from infra.job_persistence",
                    "symbol": "_write_status",
                },
                {
                    "line": 3,
                    "import": "from infra.job_persistence",
                    "symbol": "*",
                },
                {
                    "line": 6,
                    "import": "transcription record filesystem literal",
                    "symbol": "result.json",
                },
                {
                    "line": 6,
                    "import": "transcription record filesystem literal",
                    "symbol": "status.json",
                },
            ],
        },
        {
            "rule": "application_uses_public_infra_job_boundary",
            "module": "application.persistence_module",
            "path": "app/application/persistence_module.py",
            "locations": [
                {
                    "line": 4,
                    "import": "job_persistence._atomic_write_json",
                    "symbol": "_atomic_write_json",
                }
            ],
        },
        {
            "rule": "application_uses_public_infra_job_boundary",
            "module": "application.runtime_from_infra",
            "path": "app/application/runtime_from_infra.py",
            "locations": [
                {
                    "line": 4,
                    "import": "job_runtime.jobs",
                    "symbol": "jobs",
                }
            ],
        },
        {
            "rule": "application_uses_public_infra_job_boundary",
            "module": "application.runtime_module",
            "path": "app/application/runtime_module.py",
            "locations": [
                {
                    "line": 4,
                    "import": "job_runtime.jobs",
                    "symbol": "jobs",
                }
            ],
        },
    ]


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


def test_pipeline_stage_slots_do_not_static_import_provider_facades():
    """Guard source-level stage imports; runtime registry strings are separate."""

    graph = _static_internal_import_graph()
    offenders = {
        module: sorted(
            target
            for target in targets
            if target == "providers" or target.startswith("providers.")
        )
        for module, targets in graph.items()
        if module.startswith("pipeline.stages.")
        and any(
            target == "providers" or target.startswith("providers.")
            for target in targets
        )
    }

    assert offenders == {}


def test_provider_ring_does_not_static_import_pipeline_registry_or_stages():
    """Guard source-level provider imports; runtime importlib lookup is separate."""

    graph = _static_internal_import_graph()
    forbidden_prefixes = ("pipeline.registry", "pipeline.stages")
    offenders = {
        module: sorted(
            target
            for target in targets
            if any(
                target == prefix or target.startswith(f"{prefix}.")
                for prefix in forbidden_prefixes
            )
        )
        for module, targets in graph.items()
        if (module == "providers" or module.startswith("providers."))
        and any(
            target == prefix or target.startswith(f"{prefix}.")
            for prefix in forbidden_prefixes
            for target in targets
        )
    }

    assert offenders == {}


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
