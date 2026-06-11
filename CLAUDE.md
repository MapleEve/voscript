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
ignored scratch area for local-only data and ad-hoc E2E inputs
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
- If code is only for one machine, one developer, or contains private/local conventions, keep it in ignored operator config and do not commit it. Do not put secrets, hosts, tokens, or private operational notes in `CLAUDE.md`.
- Private plans, roadmaps, release strategy, and long-term planning notes belong in ignored operator-only files. Do not commit planning folders or internal roadmap documents to the public repository.

## Docs
- Update zh and en together
- Changelog: `doc/changelog.zh.md` + `doc/changelog.en.md`
- API / behavior docs must match the current implementation in `app/`; do not document fixed
  thresholds or legacy validation semantics after changing runtime behavior

## 文档与输出语言
- 本仓库后续回答、报告、ADR、规则文档和内部架构说明以中文为主；但技术证据保持原文，不翻译 `git status` 输出、文件/函数/模块名、workflow/agent 名、命令、commit ID、`grep`/`cargo`/`pytest`/test 命令名和配置 key。
- 不为了中文化而改写证据名称。例如 `app/pipeline/registry.py`、`PipelineRequest`、`source-guard test`、`import direction`、`dependency direction`、`RUST_KERNEL_MODE`、`cargo test` 这类名称按原文写。
- 会影响判断的架构术语，不能在首次出现时只写英文抽象词。一个章节或文档内首次有意义使用时，要用项目语境说明：它在 VoScript 里具体指什么、为什么重要或为什么是问题、修复后会降低什么风险。
- 需要按上一条解释的典型术语包括：`facade`、`DTO`、`owner`、`cycle`/`circular dependency`、`boundary`、`repository`、`usecase`、`orchestration`、`adapter`、`service`、`lifecycle`、`provider`、`gate`、`source-guard test`、`import direction`、`dependency direction`、`structural debt`/`architecture debt`。不要把普通命令和显而易见的工具名逐个过度解释。
- 架构重的报告或文档在有帮助时使用这个形状：`人话结论`、`架构解释`、`技术证据`。
- 本节是长期写作规则，不记录本轮进度、下一步切片或未完成状态。

## Tests
- `tests/unit/`: default regression layer for architecture and failure-path coverage
- `tests/test_security.py`: security baseline and non-live red-team regression
- `tests/e2e/`: live service validation only; require an explicit running voscript service and credentials

## CI
- Lint: `ruff check app/ --ignore E501`
- Format check: `ruff format --check app/`
- CI test slice: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/unit/ tests/test_security.py -v --tb=short --no-header`
- Full live-server validation is outside CI: use `tests/e2e/` only when a running voscript service is available
