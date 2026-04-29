# VoScript Review Guide

Use this file only as code-review guidance. Do not treat it as an implementation
plan or release checklist.

## Focus Areas

- Security and privacy: never expose tokens, hostnames, private paths, raw logs,
  job IDs, speaker IDs, or validation corpus names in code, docs, logs, PR text,
  or generated artifacts.
- Model lifecycle: look for races around lazy loading, idle unload, daemon
  shutdown, GPU semaphore ownership, CUDA cache release, and model reload after
  unload.
- GPU and CPU fallback: verify CUDA probing failures, CPU-only mode, cache-only
  model paths, and degraded alignment paths fail safely with sanitized metadata.
- API behavior: preserve existing HTTP endpoints, response contracts, optional
  fields, authentication behavior, status persistence, dedup semantics, and
  voiceprint operations unless a breaking change is explicit.
- Test coverage: prefer regression tests for failure paths, concurrency,
  persistence, security validation, and model lifecycle behavior. Do not ask for
  live E2E tests in normal PR CI.
- Documentation: keep English and Chinese docs synchronized for user-visible
  config, API, changelog, quickstart, and security behavior.

## Avoid Noise

- Do not comment on formatting that `ruff format` will handle.
- Do not request broad rewrites when a narrow fix covers the risk.
- Do not require private validation details in public docs; use public-safe
  wording such as `internal live validation` or `internal benchmark set`.
