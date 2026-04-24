#!/usr/bin/env python3
"""
AB Comparison Report: 比较 Method A 和 Method B 的转录结果。

用法：
  python3 tests/e2e/ab_comparison_report.py --a TR_ID_A --b TR_ID_B
  python3 tests/e2e/ab_comparison_report.py --a TR_ID_A --b TR_ID_B --base-url http://localhost:8780
"""
import argparse
import json
import urllib.request
import os


def get_tr(base_url, tr_id, api_key):
    req = urllib.request.Request(
        f"{base_url}/api/transcriptions/{tr_id}",
        headers={"X-API-Key": api_key},
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Method A tr_id")
    ap.add_argument("--b", required=True, help="Method B tr_id")
    ap.add_argument("--base-url", default="http://localhost:8780")
    ap.add_argument(
        "--api-key",
        default=os.getenv("VOSCRIPT_KEY") or os.getenv("VOSCRIPT_API_KEY"),
    )
    args = ap.parse_args()
    if not args.api_key:
        ap.error("--api-key or VOSCRIPT_KEY/VOSCRIPT_API_KEY is required")

    a = get_tr(args.base_url, args.a, args.api_key)
    b = get_tr(args.base_url, args.b, args.api_key)

    # Compute metrics
    a_segs = a.get("segments", [])
    b_segs = b.get("segments", [])
    a_overlap_segs = sum(1 for s in a_segs if s.get("has_overlap"))
    b_overlap_segs = sum(1 for s in b_segs if s.get("has_overlap"))
    a_text_len = sum(len(s.get("text", "")) for s in a_segs)
    b_sep_text_len = sum(
        sum(len(s.get("text", "")) for s in track.get("segments", []))
        for track in b.get("separated_tracks", [])
    )

    a_stats = a.get("overlap_stats") or {}
    b_stats = b.get("overlap_stats") or {}

    print("=" * 60)
    print("A/B COMPARISON REPORT")
    print("=" * 60)
    print(f"{'指标':<30} {'Method A':>15} {'Method B':>15}")
    print("-" * 60)
    print(f"{'Segments 总数':<30} {len(a_segs):>15} {len(b_segs):>15}")
    print(f"{'重叠 Segments 数':<30} {a_overlap_segs:>15} {b_overlap_segs:>15}")
    print(f"{'原始转录字数':<30} {a_text_len:>15} {b_sep_text_len:>15d} (sep)")
    print(
        f"{'Overlap ratio':<30} {a_stats.get('ratio', 0) * 100:>14.1f}%"
        f" {b_stats.get('ratio', 0) * 100:>14.1f}%"
    )
    print(
        f"{'Overlap 时长 (s)':<30} {a_stats.get('overlap_s', 0):>15.1f}"
        f" {b_stats.get('overlap_s', 0):>15.1f}"
    )
    print(f"{'分离音轨数':<30} {'—':>15} {len(b.get('separated_tracks', [])):>15}")
    print("=" * 60)


if __name__ == "__main__":
    main()
