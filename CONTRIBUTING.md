# 贡献指南

感谢你考虑向 voscript 贡献！

## 开始之前

- 提 issue 之前先搜一下是否已有相同问题
- 大的改动建议先开 issue 讨论设计方向

## 开发环境

```bash
# 安装依赖（需要 Python 3.11+）
pip install fastapi uvicorn pytest pytest-asyncio aiofiles httpx starlette python-multipart

# 运行测试
pytest tests/ -v

# 启动本地服务（需要 GPU + 模型文件）
cd app && uvicorn main:app --reload --port 8780
```

## 提 PR 须知

1. **Fork → 新建分支 → PR**，分支命名建议：`feat/xxx`、`fix/xxx`、`docs/xxx`
2. **测试**：新功能必须附带对应测试（`tests/` 目录），所有测试必须通过
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
pip install fastapi uvicorn pytest pytest-asyncio aiofiles httpx starlette python-multipart

# Run tests
pytest tests/ -v

# Start local server (requires GPU + model files)
cd app && uvicorn main:app --reload --port 8780
```

## Pull Request Guidelines

1. **Fork → branch → PR** — suggested branch names: `feat/xxx`, `fix/xxx`, `docs/xxx`
2. **Tests**: new features must include tests; all tests must pass
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
