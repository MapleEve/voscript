# VoScript

## Naming
- Project: **VoScript** — Docker Hub: `mapleeve/voscript` — GitHub: `MapleEve/VoScript`
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

## CI
- Run `ruff check app/ --ignore E501` before commit — zero errors required
- E2E: `pytest tests/e2e/test_api_core.py -v` — baseline 78 pass / 6 skip
