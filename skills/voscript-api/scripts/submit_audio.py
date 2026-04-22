#!/usr/bin/env python3
"""Submit an audio file to VoScript for transcription.

Usage:
    python submit_audio.py --file path/to/audio.m4a \
        [--language zh] [--min-speakers 1] [--max-speakers 4] \
        [--denoise-model default] [--snr-threshold 10.0] \
        [--no-repeat-ngram 3]

On success, prints the job ID. If the server detects that this audio has
already been transcribed (SHA-256 deduplication), it returns the existing
result immediately and this script prints a dedup notice.

Exit codes:
    0  success
    1  error (network / server / validation)
"""

from __future__ import annotations

import argparse
import mimetypes
import os
import sys
from pathlib import Path

from common import (
    VoScriptError,
    add_common_args,
    build_client_from_args,
    print_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Submit an audio file to VoScript for transcription.",
    )
    parser.add_argument(
        "--file",
        required=True,
        help="Path to the audio file to transcribe.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Optional language code hint (e.g. 'zh', 'en').",
    )
    parser.add_argument(
        "--min-speakers",
        type=int,
        default=None,
        help="Minimum speaker count hint.",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum speaker count hint.",
    )
    parser.add_argument(
        "--denoise-model",
        default=None,
        help="Denoise model identifier (server-defined).",
    )
    parser.add_argument(
        "--snr-threshold",
        type=float,
        default=None,
        help="SNR threshold (dB) for denoise gating.",
    )
    parser.add_argument(
        "--no-repeat-ngram",
        type=int,
        default=None,
        help="no_repeat_ngram_size passed to the ASR decoder.",
    )
    add_common_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    file_path = Path(args.file).expanduser()
    if not file_path.is_file():
        print(f"Error: file not found: {file_path}", file=sys.stderr)
        return 1

    form: dict[str, object] = {}
    if args.language is not None:
        form["language"] = args.language
    if args.min_speakers is not None:
        form["min_speakers"] = args.min_speakers
    if args.max_speakers is not None:
        form["max_speakers"] = args.max_speakers
    if args.denoise_model is not None:
        form["denoise_model"] = args.denoise_model
    if args.snr_threshold is not None:
        form["snr_threshold"] = args.snr_threshold
    if args.no_repeat_ngram is not None:
        form["no_repeat_ngram_size"] = args.no_repeat_ngram

    mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

    try:
        client = build_client_from_args(args)
        with file_path.open("rb") as fh:
            files = {"file": (file_path.name, fh, mime)}
            response = client.post("/api/transcribe", data=form, files=files)
    except (VoScriptError, ValueError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not isinstance(response, dict):
        print("Error: unexpected response shape from /api/transcribe", file=sys.stderr)
        print_json(response)
        return 1

    job_id = response.get("id", "<unknown>")
    status = response.get("status", "<unknown>")
    deduplicated = bool(response.get("deduplicated"))

    if deduplicated:
        print(f"Audio already transcribed (SHA-256 dedup). Job ID: {job_id}")
    else:
        print(f"Transcription queued. Job ID: {job_id}")

    print(f"Status: {status}")
    print_json(response)
    return 0


if __name__ == "__main__":
    sys.exit(main())
