#!/usr/bin/env python3
"""Scan tracked files for public-release privacy leaks."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

MEDIA_EXTENSIONS = {
    ".aac",
    ".flac",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
}
TEXT_EXTENSIONS = {
    ".dockerignore",
    ".env",
    ".gitignore",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".rst",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ALLOW_CONTENT = {"voscript-api/scripts/public_release_scan.py"}
PUBLIC_CONTAINER_PATHS = {
    "/data/voiceprints",
    "/data/transcriptions/asnorm_cohort.npy",
}


@dataclass(frozen=True)
class Finding:
    category: str
    path: str
    line: int
    excerpt: str
    advice: str


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    advice: str


def _rx(value: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(value, flags)


SECRET_ASSIGNMENT_RE = _rx(
    r"(?<![A-Za-z0-9_-])(?:export\s+)?['\"]?(?P<name>[A-Za-z_][A-Za-z0-9_-]*)['\"]?"
    r"\s*(?:=|:)\s*(?P<value>\"[^\"\n]*\"|'[^'\n]*'|`[^`\n]*`|[^\s,;]+)?",
    re.I,
)
SECRET_VALUE_PREFIX_RE = _rx(
    r"^(?:sk[-_](?:(?:live|test)[-_])?|hf_|ghp_|github_pat_|xox[baprs]-|AKIA|AIza|eyJ)"
    r"[A-Za-z0-9_+/\-.=]{8,}$"
)
RAW_BEARER_TOKEN_RE = _rx(
    r"(?<![A-Za-z0-9_-])Bearer\s+(?P<value>[A-Za-z0-9_+/\-.=]{12,})(?![A-Za-z0-9_-])",
    re.I,
)
RAW_SECRET_TOKEN_RE = _rx(
    r"(?<![A-Za-z0-9_+/\-.=])"
    r"(?P<value>(?:sk[-_](?:(?:live|test)[-_])?|hf_|ghp_|github_pat_|xox[baprs]-|AKIA|AIza|eyJ)"
    r"[A-Za-z0-9_+/\-.=]{8,})"
    r"(?![A-Za-z0-9_+/\-.=])"
)
PLACEHOLDER_VALUE_RE = _rx(
    r"^(?:<[^>\s]+>|\$\{[A-Za-z_][A-Za-z0-9_]*(?::-[^}]*)?\}|\$[A-Za-z_][A-Za-z0-9_]*|"
    r"(?:your|example|sample|dummy|fake|test)[-_]?[A-Za-z0-9_-]*"
    r"(?:api[-_]?key|key|token|password|secret)[A-Za-z0-9_-]*)$",
    re.I,
)
EXPLICIT_SECRET_NAMES = {
    "api_key",
    "hf_token",
    "password",
    "secret",
    "token",
    "voscript_api_key",
    "voscript_token",
}

PRIVATE_DIRECT_SSH_ALIAS = "a" + "i"
PRIVATE_WAN_SSH_ALIAS = PRIVATE_DIRECT_SSH_ALIAS + "-" + "wan"
PRIVATE_REMOTE_ALIASES = (PRIVATE_DIRECT_SSH_ALIAS + "-" + "lan", PRIVATE_WAN_SSH_ALIAS)
PRIVATE_PROXY_HOST_PORT = "127" + ".0.0.1:" + "78" + "97"

LINE_RULES = [
    Rule(
        "private validation corpus name",
        _rx(r"\b(?:private E2E (?:corpus|sample)|private (?:corpus|sample))\b", re.I),
        "Use anonymized wording such as internal live validation.",
    ),
    Rule(
        "machine-local path",
        _rx(r"(?:/Users/|/data/)[^\s)\"'`]+"),
        "Move machine-specific paths to local-only notes.",
    ),
    Rule(
        "local-only directory reference",
        _rx(r"\b(?:roadmap|tmp)[/\\]"),
        "Replace local-only directory paths with generic ignored archive wording.",
    ),
    Rule(
        "local ssh config path",
        _rx(r"(?:~[/\\])?\.ssh[/\\]config"),
        "Refer to local SSH config generically in public docs.",
    ),
    Rule(
        "remote host alias",
        _rx(
            rf"(?:\bssh\s+{PRIVATE_DIRECT_SSH_ALIAS}\b|`{PRIVATE_DIRECT_SSH_ALIAS}`|\b(?:{'|'.join(PRIVATE_REMOTE_ALIASES)})\b)",
            re.I,
        ),
        "Replace private host aliases with generic deployment wording.",
    ),
    Rule(
        "private proxy/debug port",
        _rx(re.escape(PRIVATE_PROXY_HOST_PORT)),
        "Replace private proxy/debug ports with placeholders.",
    ),
    Rule(
        "candidate debug port",
        _rx(r"\b18(?:7[0-9]{2}|78[0-9])\b"),
        "Do not publish temporary candidate/debug ports.",
    ),
    Rule(
        "real transcription id",
        _rx(r"\btr_[0-9]{8}_[0-9]{6}_[A-Za-z0-9_-]+\b"),
        "Replace real transcription IDs with <tr_id>.",
    ),
    Rule(
        "real speaker id",
        _rx(r"\bspk_[0-9a-f]{6,}\b", re.I),
        "Replace real speaker IDs with <speaker_id>.",
    ),
]


def run_git(root: Path, args: Sequence[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def tracked_files(root: Path) -> list[Path]:
    proc = run_git(root, ["ls-files", "-z"])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip())
    return [Path(p.decode("utf-8")) for p in proc.stdout.split(b"\0") if p]


def is_text_file(path: Path) -> bool:
    if path.name in {".gitignore", ".dockerignore"}:
        return True
    return path.suffix.lower() in TEXT_EXTENSIONS or not path.suffix


def path_findings(paths: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for rel in paths:
        rel_str = rel.as_posix()
        suffix = rel.suffix.lower()
        if rel.parts and rel.parts[0] in {"roadmap", "tmp"}:
            findings.append(
                Finding(
                    "tracked local-only directory",
                    rel_str,
                    1,
                    rel_str,
                    "Remove tracked planning/tmp artifacts or move them to ignored local storage.",
                )
            )
        if rel.name == "CLAUDE.local.md" or (
            rel.name.startswith(".env") and rel.name != ".env.example"
        ):
            findings.append(
                Finding(
                    "tracked local config file",
                    rel_str,
                    1,
                    rel_str,
                    "Untrack local config files and keep them ignored.",
                )
            )
        if suffix in MEDIA_EXTENSIONS:
            findings.append(
                Finding(
                    "tracked media corpus file",
                    rel_str,
                    1,
                    rel_str,
                    "Do not publish raw audio/video validation material.",
                )
            )
        if suffix in {".log", ".json", ".txt"} and "validation" in rel_str.lower():
            findings.append(
                Finding(
                    "tracked validation artifact",
                    rel_str,
                    1,
                    rel_str,
                    "Keep raw validation logs/results local-only.",
                )
            )
    return findings


def is_secret_name(name: str) -> bool:
    normalized = name.replace("-", "_").lower()
    return (
        normalized in EXPLICIT_SECRET_NAMES
        or "api_key" in normalized
        or normalized.endswith("_token")
        or normalized == "token"
        or "password" in normalized
        or "secret" in normalized
    )


def assigned_value(raw: str | None) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    if value[0] in {"'", '"', "`"}:
        quote = value[0]
        end = value.find(quote, 1)
        if end == -1:
            return value[1:].strip()
        return value[1:end].strip()
    return (
        value.split("#", 1)[0].strip().split(maxsplit=1)[0].rstrip("`.,;)]\"'").strip()
    )


def is_placeholder_secret_value(value: str) -> bool:
    normalized = value.strip()
    if normalized in {"", "..."}:
        return True
    if normalized.lower() in {
        "changeme",
        "change-me",
        "change_me",
        "placeholder",
        "redacted",
        "replace-me",
        "replace_me",
        "todo",
    }:
        return True
    return bool(PLACEHOLDER_VALUE_RE.fullmatch(normalized))


def looks_like_secret_value(value: str) -> bool:
    if is_placeholder_secret_value(value):
        return False
    compact = re.sub(r"\s+", "", value.strip())
    if SECRET_VALUE_PREFIX_RE.match(compact):
        return True
    if len(compact) < 12:
        return False
    if len([char for char in compact if char.isalnum()]) < 12:
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_+/\-.=]+", compact))


def is_backtick_wrapped_value(raw: str | None) -> bool:
    return bool(raw and raw.strip().startswith("`"))


def _excerpt(line: str) -> str:
    excerpt = line.strip()
    return excerpt if len(excerpt) <= 180 else excerpt[:177] + "..."


def secret_assignment_finding(path: str, line_no: int, line: str) -> Finding | None:
    for match in SECRET_ASSIGNMENT_RE.finditer(line):
        if not is_secret_name(match.group("name")):
            continue
        raw_value = match.group("value")
        value = assigned_value(raw_value)
        if not value or is_placeholder_secret_value(value):
            continue
        if not looks_like_secret_value(value) and not is_backtick_wrapped_value(
            raw_value
        ):
            continue
        return Finding(
            "secret-looking assignment",
            path,
            line_no,
            _excerpt(line),
            "Use placeholders such as <API_KEY>; rotate if this is a real secret.",
        )
    return None


def raw_secret_token_finding(path: str, line_no: int, line: str) -> Finding | None:
    for pattern in (RAW_BEARER_TOKEN_RE, RAW_SECRET_TOKEN_RE):
        for match in pattern.finditer(line):
            if looks_like_secret_value(assigned_value(match.group("value"))):
                return Finding(
                    "secret-looking raw token",
                    path,
                    line_no,
                    _excerpt(line),
                    "Use placeholders such as <API_KEY>; rotate if this is a real secret.",
                )
    return None


def line_findings(root: Path, paths: Iterable[Path]) -> list[Finding]:
    findings: list[Finding] = []
    for rel in paths:
        rel_str = rel.as_posix()
        if rel.name in {".gitignore", ".npmignore", ".dockerignore"}:
            continue
        if rel_str in ALLOW_CONTENT or not is_text_file(rel):
            continue
        try:
            lines = (
                (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
            )
        except OSError:
            continue
        for line_no, line in enumerate(lines, start=1):
            for maybe_finding in (
                secret_assignment_finding(rel_str, line_no, line),
                raw_secret_token_finding(rel_str, line_no, line),
            ):
                if maybe_finding is not None:
                    findings.append(maybe_finding)
            for rule in LINE_RULES:
                if not rule.pattern.search(line):
                    continue
                if rule.name == "machine-local path" and any(
                    path in line for path in PUBLIC_CONTAINER_PATHS
                ):
                    continue
                findings.append(
                    Finding(rule.name, rel_str, line_no, _excerpt(line), rule.advice)
                )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan Git-tracked files for VoScript public-release privacy leaks."
    )
    parser.add_argument("--root", default=".", help="Repository root to scan.")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"root does not exist: {root}", file=sys.stderr)
        return 2

    try:
        paths = tracked_files(root)
    except RuntimeError as exc:
        print(f"not a git worktree or git failed: {exc}", file=sys.stderr)
        return 2

    findings = path_findings(paths) + line_findings(root, paths)
    if findings:
        print("Public release scan failed:")
        for item in findings:
            print(f"- {item.path}:{item.line}: {item.category}")
            print(f"  {item.excerpt}")
            print(f"  {item.advice}")
        return 1

    print(f"Public release scan passed ({len(paths)} tracked files).")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    sys.exit(main())
