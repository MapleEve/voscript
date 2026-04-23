# VoScript

## Naming
- Project: **VoScript** — Docker Hub: `mapleeve/voscript` — GitHub: `MapleEve/voscript`
- Integration client: **BetterAINote**
- License: **Custom — free for individuals, written authorization required for commercial use**

## Versioning
- Format: `MAJOR.MINOR.PATCH` — bump patch for fixes/small additions, minor for new features, major for breaking changes
- Version in `app/main.py` and `doc/changelog.*.md` must stay in sync

## Structure
```
app/             FastAPI service (Docker image)
doc/             Docs — every file has zh + en counterpart
tests/e2e/       E2E tests against live server
.github/workflows/
  ci.yml         lint + test + security-scan
  release.yml    Docker build+push on tag or workflow_dispatch
```

## Docs
- Update zh and en together
- Changelog: `doc/changelog.zh.md` + `doc/changelog.en.md`
- API / behavior docs must match the current implementation in `app/`; do not document fixed
  thresholds or legacy validation semantics after changing runtime behavior

## CI
- Lint: `ruff check app/ --ignore E501`
- Format check: `ruff format --check app/`
- CI test slice: `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/unit/ tests/test_security.py -v --tb=short --no-header`
- Full live-server validation is outside CI: use `tests/e2e/` only when a running voscript service is available
