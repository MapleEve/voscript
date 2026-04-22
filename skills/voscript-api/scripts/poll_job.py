#!/usr/bin/env python3
"""Poll a VoScript transcription job until it completes or fails.

Usage:
    python poll_job.py --job-id tr_xxx [--interval 5.0] [--timeout 600]

Status progression (from the VoScript FSM):
    queued -> converting -> denoising -> transcribing -> identifying -> completed

Prints one line per status change. Exits 0 on ``completed``, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
import time

from common import (
    VoScriptError,
    add_common_args,
    build_client_from_args,
)


STATUS_MESSAGES = {
    "queued": "Waiting...",
    "converting": "Processing audio...",
    "denoising": "Denoising...",
    "transcribing": "Transcribing...",
    "identifying": "Identifying speakers...",
    "completed": "Done!",
    "failed": "Failed.",
    "error": "Error.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Poll a VoScript transcription job until completion.",
    )
    parser.add_argument(
        "--job-id",
        required=True,
        help="Job ID returned by submit_audio (format: tr_xxx).",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds (default: 5.0).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Max total wait time in seconds (default: 600).",
    )
    add_common_args(parser)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        client = build_client_from_args(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    deadline = time.monotonic() + max(1, args.timeout)
    interval = max(0.1, args.interval)
    last_status: str | None = None

    while True:
        try:
            job = client.get(f"/api/jobs/{args.job_id}")
        except VoScriptError as exc:
            print(f"Error fetching job: {exc}", file=sys.stderr)
            return 1

        if not isinstance(job, dict):
            print("Error: unexpected /api/jobs response shape", file=sys.stderr)
            return 1

        status = str(job.get("status", "unknown"))
        if status != last_status:
            hint = STATUS_MESSAGES.get(status, "")
            tr_id = (
                job.get("result", {}).get("id")
                if isinstance(job.get("result"), dict)
                else None
            )
            if status == "completed" and tr_id:
                print(f"[{status}] {hint} tr_id: {tr_id}")
            else:
                print(f"[{status}] {hint}".rstrip())
            last_status = status

        if status == "completed":
            return 0
        if status in {"failed", "error"}:
            err = job.get("error") or "(no error message returned)"
            print(f"Job ended with status={status}: {err}", file=sys.stderr)
            return 1

        if time.monotonic() >= deadline:
            print(
                f"Timeout after {args.timeout}s while waiting for job {args.job_id} "
                f"(last status: {status}).",
                file=sys.stderr,
            )
            return 1

        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
