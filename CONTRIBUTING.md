# 贡献指南

感谢你考虑向 voscript 贡献！

## 开始之前

- 提 issue 之前先搜一下是否已有相同问题
- 大的改动建议先开 issue 讨论设计方向

## 开发环境

```bash
# 需要 Python 3.11+

# 轻量 lint / unit / security 流程（与当前 CI 对齐）
pip install ruff pytest pytest-cov fastapi httpx numpy aiofiles starlette python-multipart
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/unit/ tests/test_security.py -v --tb=short --no-header
ruff check app/ --ignore E501
ruff format --check app/

# 需要真实服务 / GPU 时，再安装完整运行时依赖并启动
pip install -r app/requirements.txt
(cd app && uvicorn main:app --reload --port 8780)

# 另开一个 shell，在仓库根目录跑 e2e
pytest tests/e2e/test_api_core.py -v
```

## 提 PR 须知

1. **Fork → 新建分支 → PR**，分支命名建议：`feat/xxx`、`fix/xxx`、`docs/xxx`
2. **测试**：新功能必须附带对应测试（`tests/` 目录）。至少通过当前 CI 的 lint + format + `tests/unit/` + `tests/test_security.py`；涉及真实转录链路时再补跑对应 `tests/e2e/`
3. **文档**：影响 API 或行为的改动需同步更新 `doc/` 中的相关文档
4. **提交信息**：用英文小写动词开头，例如 `fix: handle zero-length audio` / `feat: add speaker rename API`

## 代码风格

- Python：PEP 8，函数命名 `snake_case`，类型注解尽量完整
- 不要引入 print 调试语句，用 `logger.debug()`
- 安全敏感代码（鉴权、文件路径）改动务必在 PR 描述中说明

## 报告安全问题

**不要开公开 issue**。请直接发 email 或通过 GitHub Security Advisory 私下报告。
详见 [SECURITY.md](./SECURITY.md)。

## 行为准则

友善、建设性地交流。维护者有权关闭不遵守基本礼仪的讨论。

---

# Contributing (English)

Thank you for contributing to voscript!

## Before You Start

- Search existing issues before filing a new one
- For large changes, open an issue first to discuss design direction

## Development Setup

```bash
# Requires Python 3.11+

# Lightweight lint / unit / security slice (matches current CI)
pip install ruff pytest pytest-cov fastapi httpx numpy aiofiles starlette python-multipart
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/unit/ tests/test_security.py -v --tb=short --no-header
ruff check app/ --ignore E501
ruff format --check app/

# Full runtime for local service / GPU-backed validation
pip install -r app/requirements.txt
(cd app && uvicorn main:app --reload --port 8780)

# In another shell, from the repo root, run E2E
pytest tests/e2e/test_api_core.py -v
```

## Pull Request Guidelines

1. **Fork → branch → PR** — suggested branch names: `feat/xxx`, `fix/xxx`, `docs/xxx`
2. **Tests**: new features must include tests. At minimum pass the current CI slice (lint + format + `tests/unit/` + `tests/test_security.py`); run `tests/e2e/` as well when changing the real transcription pipeline
3. **Docs**: changes to API behavior must update the relevant files in `doc/`
4. **Commit messages**: lowercase verb prefix, e.g. `fix: handle zero-length audio`

## Code Style

- Python: PEP 8, `snake_case` for functions, type annotations where practical
- Use `logger.debug()` instead of `print`
- Security-sensitive changes (auth, file paths) must be explained in the PR description

## Reporting Security Issues

**Do not open a public issue.** Use GitHub Security Advisory or email privately.
See [SECURITY.md](./SECURITY.md).

## Code of Conduct

Be kind and constructive. Maintainers may close discussions that violate basic courtesy norms.
