#!/usr/bin/env python3
"""Export a VoScript transcription as SRT / TXT / JSON.

Usage:
    python export_transcript.py --tr-id tr_xxx \
        [--format srt|txt|json] [--output path/to/file]

If ``--output`` is omitted, the exported content is printed to stdout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from common import (
    VoScriptError,
    add_common_args,
    build_client_from_args,
)


SUPPORTED_FORMATS = ("srt", "txt", "json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a VoScript transcription in srt/txt/json format.",
    )
    parser.add_argument(
        "--tr-id",
        required=True,
        help="Transcription ID (format: tr_xxx).",
    )
    parser.add_argument(
        "--format",
        choices=SUPPORTED_FORMATS,
        default="srt",
        help="Export format (default: srt).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output file path. If omitted, content is printed to stdout.",
    )
    add_common_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        client = build_client_from_args(args)
        content, _suggested = client.download(
            f"/api/export/{args.tr_id}",
            params={"format": args.format},
        )
    except (VoScriptError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if args.output:
        out_path = Path(args.output).expanduser()
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(content)
        except OSError as exc:
            print(f"Error writing {out_path}: {exc}", file=sys.stderr)
            return 1
        print(f"Wrote {len(content)} bytes to {out_path}")
        return 0

    try:
        sys.stdout.write(content.decode("utf-8"))
    except UnicodeDecodeError:
        sys.stdout.buffer.write(content)
    if not content.endswith(b"\n"):
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
