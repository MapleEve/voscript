#!/usr/bin/env python3
"""Fetch a completed VoScript transcription and print it in a readable form.

Usage:
    python fetch_result.py --tr-id tr_xxx [--show-words]

Prints each segment as:
    [HH:MM:SS.mmm - HH:MM:SS.mmm] speaker_name: text

Then prints a speaker_map summary and a note about similarity semantics.

Note:
    Similarity values in VoScript are AS-norm z-scores, NOT [0,1] probabilities.
    Higher z-score = more confident match against the cohort distribution.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

from common import (
    VoScriptError,
    add_common_args,
    build_client_from_args,
    format_hms,
    print_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch and display a VoScript transcription result.",
    )
    parser.add_argument(
        "--tr-id",
        required=True,
        help="Transcription ID (format: tr_xxx).",
    )
    parser.add_argument(
        "--show-words",
        action="store_true",
        help="Include word-level alignment under each segment.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the raw JSON response instead of the formatted view.",
    )
    add_common_args(parser)
    return parser


def _segment_speaker(seg: dict[str, Any], speaker_map: dict[str, Any]) -> str:
    label = seg.get("speaker") or seg.get("speaker_label")
    if not label:
        return "unknown"
    mapped = speaker_map.get(label) if isinstance(speaker_map, dict) else None
    if isinstance(mapped, dict):
        return str(mapped.get("name") or mapped.get("speaker_id") or label)
    if isinstance(mapped, str):
        return mapped
    return str(label)


def _print_segments(result: dict[str, Any], show_words: bool) -> None:
    segments = result.get("segments") or []
    speaker_map = result.get("speaker_map") or {}

    if not segments:
        print("(no segments in result)")
        return

    for seg in segments:
        start = format_hms(seg.get("start"))
        end = format_hms(seg.get("end"))
        speaker = _segment_speaker(seg, speaker_map)
        text = (seg.get("text") or "").strip()
        seg_id = seg.get("id")
        prefix = f"[{start} - {end}]"
        if seg_id is not None:
            prefix = f"#{seg_id} {prefix}"
        print(f"{prefix} {speaker}: {text}")

        if show_words:
            words = seg.get("words") or []
            for w in words:
                ws = format_hms(w.get("start"))
                we = format_hms(w.get("end"))
                token = (w.get("word") or w.get("text") or "").strip()
                print(f"    {ws}-{we} {token}")


def _print_speaker_map(result: dict[str, Any]) -> None:
    speaker_map = result.get("speaker_map") or {}
    if not speaker_map:
        print("\nSpeaker map: (empty)")
        return

    print("\nSpeaker map:")
    for label, info in speaker_map.items():
        if isinstance(info, dict):
            name = info.get("name") or "(unnamed)"
            speaker_id = info.get("speaker_id") or "-"
            similarity = info.get("similarity")
            sim_str = (
                f" similarity={similarity:.3f}"
                if isinstance(similarity, (int, float))
                else ""
            )
            print(f"  {label} -> {name} (speaker_id={speaker_id}){sim_str}")
        else:
            print(f"  {label} -> {info}")

    print(
        "\nNote: similarity values above are AS-norm z-scores, not probabilities. "
        "Higher z-score = stronger match vs. cohort distribution."
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        client = build_client_from_args(args)
        result = client.get(f"/api/transcriptions/{args.tr_id}")
    except (VoScriptError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if not isinstance(result, dict):
        print("Error: unexpected response from /api/transcriptions", file=sys.stderr)
        return 1

    if args.json:
        print_json(result)
        return 0

    filename = result.get("filename") or "(unknown)"
    created = result.get("created_at") or "-"
    print(f"Transcription: {args.tr_id}")
    print(f"  filename:   {filename}")
    print(f"  created_at: {created}")
    seg_count = len(result.get("segments") or [])
    spk_count = len(result.get("speaker_map") or {})
    print(f"  segments:   {seg_count}")
    print(f"  speakers:   {spk_count}")
    print("")

    _print_segments(result, show_words=args.show_words)
    _print_speaker_map(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
