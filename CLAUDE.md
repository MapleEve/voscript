# VoScript

## Naming
- Project: **VoScript** — Docker Hub: `mapleeve/voscript` — GitHub: `MapleEve/voscript`
- Integration client: **BetterAINote**
- License: **Custom — free for individuals, written authorization required for commercial use**

## Versioning
- Format: `MAJOR.MINOR.PATCH` — bump patch for fixes/small additions, minor for new features, major for breaking changes
- Version in `app/main.py` and `doc/changelog.*.md` must stay in sync

## Repo Layout
```text
app/        runtime code
doc/        user-facing docs and changelogs
tests/      unit, security, and live E2E tests
tmp/        local scratch data and ad-hoc E2E inputs
```

## App Layout
```text
app/
  api/
    deps.py
    routers/
  application/
  pipeline/
    contracts/
    stages/
    registry.py
    runner.py
    orchestrator.py
  providers/
  voiceprints/
  infra/
  main.py
  config.py
  static/
```

## Placement Rules
- `app/api/`: FastAPI entrypoints only. Put request parsing, response shaping, auth/dependency wiring, and HTTP error mapping here. Do not put pipeline logic, job state machines, or file persistence here.
- `app/application/`: Use-case orchestration. Put transcription job flow, status transitions, dedup orchestration, and cross-module coordination here. Do not put FastAPI handlers or model-specific code here.
- `app/pipeline/`: Stable processing flow. Put request/context/result contracts, stage order, stage dispatch, runner logic, and orchestration boundaries here.
- `app/pipeline/stages/`: One directory per stable step. Stage code should be thin and delegate step-specific work to providers.
- `app/providers/`: Backend/model implementations for a pipeline step. Use `app/providers/<step>/<impl>.py`. Canonical step names are `ingest`, `normalize`, `enhance`, `vad`, `asr`, `diarization`, `embedding`, `voiceprint_match`, `punc`, `postprocess`, and `artifacts`.
- `app/voiceprints/`: Voiceprint domain logic only. Put enrollment, matching, cohort rebuild, scoring, repository, and storage abstractions here.
- `app/infra/`: Concrete adapters only. Put filesystem writes, artifact persistence, temp file cleanup, path safety, hashes, job persistence, and runtime helpers here.
- `app/main.py`: Composition root and app lifecycle only.
- `app/config.py`: Environment/config definitions only.
- `app/static/`: Static frontend assets only.

## Structure Rules
- Prefer canonical step names `normalize` and `enhance`; do not introduce new `input_normalization` or `enhancement` modules.
- Add new provider implementations under the existing step directory instead of branching logic inside routers, application code, or the runner.
- Do not reintroduce flat legacy modules such as `app/pipeline.py`, `app/voiceprint_db.py`, or `app/services/*`.
- If code is only for one machine, one developer, or contains private/local conventions, keep it in `CLAUDE.local.md` and do not commit it. Do not put secrets, hosts, tokens, or private operational notes in `CLAUDE.md`.
- Private plans, roadmaps, release strategy, and long-term planning notes belong in `CLAUDE.local.md` or other gitignored local files. Do not commit planning folders or internal roadmap documents to the public repository.

## Docs
- Update zh and en together
- Changelog: `doc/changelog.zh.md` + `doc/changelog.en.md`
- API / behavior docs must match the current implementation in `app/`; do not document fixed
  thresholds or legacy validation semantics after changing runtime behavior

## Tests
- `tests/unit/`: default regression layer for architecture and failure-path coverage
- `tests/test_security.py`: security baseline and non-live red-team regression
- `tests/e2e/`: live service validation only; require an explicit running voscript service and credentials

## CI
- Lint: `ruff check app/ --ignore E501`
- Format check: `ruff format --check app/`
- CI test slice: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/unit/ tests/test_security.py -v --tb=short --no-header`
- Full live-server validation is outside CI: use `tests/e2e/` only when a running voscript service is available
