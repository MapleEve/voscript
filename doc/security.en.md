# Security Policy

[简体中文](./security.zh.md) | **English**

## Supported Versions

Only `main` is supported. Run the latest published image or rebuild
from `main`.

## Threat Model

This service uploads audio, runs speaker diarization, and stores
speaker voiceprints. It is designed for **trusted deployments**. By
default, any client that can reach `:8780` and holds `API_KEY` can:

- Upload audio into `data/uploads/` and trigger GPU inference
- Read every transcript under `data/transcriptions/`
- Manipulate every enrolled voiceprint (persistent speaker embeddings
  — biometric data)

Treat the service as if it were an internal database.

## Built-in hardening (on by default)

As of 0.6.0 the following protections are in place out of the box:

1. **Container runs as a non-root user.** The Dockerfile creates an
   `app` user (uid/gid 1000 by default, overridable via `APP_UID`/
   `APP_GID`) and ends with `USER app`. A code-execution bug inside
   the service only grants that low-privilege account — it cannot
   read root-owned files on the host.
2. **Upload size limit via `MAX_UPLOAD_BYTES`** (default 2 GiB).
   Chunked streaming copy aborts with `413` on overflow and deletes
   the partial artifact — no disk-exhaustion DoS.
3. **Upload filename sanitization.** Only the final path component
   of the client-supplied `filename` is kept. `../../etc/passwd.wav`
   reduces to `passwd.wav` before the save path is built.
4. **`--` inserted before the ffmpeg input path.** Closes option
   parsing so a filename like `-Y.mp4` can't be interpreted as a flag.
5. **Constant-time key comparison.** `hmac.compare_digest` instead of
   `!=` removes any timing side channel.
6. **Atomic, locked voiceprint DB.** SQLite WAL mode provides atomic
   writes at the database level; a process-level `threading.RLock`
   serializes concurrent mutations so parallel enroll/delete operations
   never corrupt the store.
7. **`np.load(..., allow_pickle=False)` everywhere.** Closes the
   `torch.load`-style pickle RCE path.
8. **Exact-match `/docs`, `/redoc`, `/openapi.json`.** The previous
   `startswith("/docs")` let `/docsXYZ` slip past middleware; now it
   correctly returns 401.
9. **Path traversal protection**: `safe_tr_dir()` validates `tr_id` with regex `^tr_[A-Za-z0-9_-]{1,64}$` + `resolve()` prefix check; `safe_speaker_label()` applies equivalent character set restrictions
10. **Log injection prevention**: `safe_log_filename()` strips control characters from user-supplied filenames before they reach log lines
11. **Route parameter validation**: FastAPI `Path(pattern=...)` rejects malformed IDs at the framework level
12. **ffmpeg timeout**: `FFMPEG_TIMEOUT_SEC` (default 1800 s) prevents malformed audio from hanging the process
13. **Pickle protection**: `np.load(allow_pickle=False)` prevents arbitrary code execution from malicious `.npy` embedding files
14. **Zero-vector defense**: voiceprint `identify()` returns early on all-zero embeddings, preventing AS-norm scoring from producing false matches

## Required deployment-side hardening

Things the code can't enforce that the operator must get right:

1. **Set `API_KEY`.** Without it the service accepts unauthenticated
   requests and logs a startup warning. Any deployment not on a fully
   trusted LAN segment MUST set this env var to a long random string.
   Clients send it as `Authorization: Bearer <key>` or
   `X-API-Key: <key>`.
   - Generate one: `openssl rand -hex 32`
   - If you are genuinely on a trusted internal network and do not need authentication, set `ALLOW_NO_AUTH=1` to suppress the startup warning (this variable provides no authentication — it only declares "I understand the implications of running without auth").
2. **Never commit `.env`.** Only `.env.example` belongs in git.
3. **Do not expose `:8780` to the public Internet.** Put it behind a
   VPN, a reverse proxy with TLS, or at minimum an IP allow-list.
   `API_KEY` alone is not a substitute for transport encryption.
4. **Keep your HuggingFace token out of logs and out of images.** It
   is read from `HF_TOKEN` at runtime and used only to download
   pyannote models — nothing else.
5. **Back up `data/voiceprints/`.** Voiceprints are biometric data;
   losing them means re-enrolling every speaker, and leaking them is
   worse than leaking regular DB rows.
6. **Match the host directory owner to `APP_UID`/`APP_GID`.** The
   container runs as uid 1000 by default — if your `DATA_DIR` is
   owned by a different user, either `chown -R 1000:1000 DATA_DIR`
   or set `APP_UID`/`APP_GID` in `.env` to the real owner.

## Known limitations / not covered

- **No built-in rate limiting or failed-auth lockout.** Acceptable in
  the single-tenant + long-random-key threat model; once the key
  leaks, brute-force isn't throttled. Put a rate limit at the
  reverse-proxy layer if you need one.
- **No TLS terminated by this service.** Intentional — the service is
  meant to sit behind nginx/caddy/traefik. Only expose `:8780` on a
  network you trust.
- **`server: uvicorn` response header.** Minor fingerprint disclosure,
  no real attack surface, not scrubbed.

## Reporting a vulnerability

Please open a private security advisory on GitHub or email the
maintainer. **Do not** file public issues for unpatched
vulnerabilities.
