"""Overlap bench: upload a private E2E corpus, wait for
transcription, compute per-file overlap statistics.

Overlap = time intervals where two or more segments from different speakers
are simultaneously active.  The metric answers "how much cross-talk did the
diariser detect?"

Usage:
  python tests/e2e/bench_overlap.py [--bench-dir PATH] [--out-json PATH]

Env vars (same as test_api_core.py):
  VOSCRIPT_URL  (default: http://localhost:8780)
  VOSCRIPT_KEY  or VOSCRIPT_API_KEY
  BENCH_DIR     (default: tmp/private_e2e_corpus)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BASE_URL = os.getenv("VOSCRIPT_URL", "http://localhost:8780").rstrip("/")
API_KEY = os.getenv("VOSCRIPT_KEY") or os.getenv("VOSCRIPT_API_KEY") or ""
POLL_INTERVAL = 15  # seconds
POLL_TIMEOUT = 1800  # 30 min per file (long meetings)
_NO_PROXY = {"http": None, "https": None}

BENCH_DIR = Path(os.getenv("BENCH_DIR", "tmp/private_e2e_corpus"))


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _hdrs():
    if not API_KEY:
        raise RuntimeError("VOSCRIPT_KEY or VOSCRIPT_API_KEY is required")
    return {"X-API-Key": API_KEY}


def _get(path):
    return requests.get(BASE_URL + path, headers=_hdrs(), timeout=30, proxies=_NO_PROXY)


def _upload(file_path: Path) -> dict:
    with open(file_path, "rb") as fh:
        resp = requests.post(
            BASE_URL + "/api/transcribe",
            headers=_hdrs(),
            data={"language": "zh"},
            files={"file": (file_path.name, fh, "audio/ogg")},
            timeout=120,
            proxies=_NO_PROXY,
        )
    resp.raise_for_status()
    return resp.json()


def _poll(job_id: str) -> dict:
    deadline = time.time() + POLL_TIMEOUT
    dots = 0
    while time.time() < deadline:
        time.sleep(POLL_INTERVAL)
        resp = _get(f"/api/jobs/{job_id}")
        resp.raise_for_status()
        data = resp.json()
        status = data["status"]
        dots += 1
        print(f"  [{dots * POLL_INTERVAL}s] {status}", flush=True)
        if status == "completed":
            if data.get("result"):
                return data["result"]
            tr = _get(f"/api/transcriptions/{job_id}")
            tr.raise_for_status()
            return tr.json()
        if status == "failed":
            raise RuntimeError(f"Job {job_id} failed: {data.get('error')}")
    raise TimeoutError(f"Job {job_id} timed out after {POLL_TIMEOUT}s")


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------


def compute_overlap_stats(segments: list) -> dict:
    """Compute overlap statistics from a list of diarisation segments.

    Each segment: {start, end, speaker_label, ...}

    Returns a dict with:
      total_duration   - end of last segment (seconds)
      overlap_duration - total seconds where ≥2 speakers are active
      overlap_ratio    - overlap_duration / total_duration
      n_segments       - total segment count
      n_speakers       - unique speaker count
      overlap_events   - list of {start, end, speakers} overlapping intervals
    """
    if not segments:
        return {
            "total_duration": 0,
            "overlap_duration": 0,
            "overlap_ratio": 0,
            "n_segments": 0,
            "n_speakers": 0,
            "overlap_events": [],
        }

    # Build (start, end, speaker) tuples
    turns = [
        (
            float(s["start"]),
            float(s["end"]),
            s.get("speaker_label", s.get("speaker", "?")),
        )
        for s in segments
        if float(s["end"]) > float(s["start"])
    ]

    total_duration = max(e for _, e, _ in turns) if turns else 0
    speakers = sorted({sp for _, _, sp in turns})

    # Sweep-line: find all intervals where ≥2 speakers active simultaneously
    events = []  # (time, +1/-1, speaker)
    for start, end, sp in turns:
        events.append((start, +1, sp))
        events.append((end, -1, sp))

    events.sort(key=lambda x: (x[0], x[1]))  # sort by time; end before start at same t

    overlap_duration = 0.0
    overlap_events = []
    active: dict[str, int] = {}  # speaker -> nested depth (usually 0/1)
    prev_time = None

    for t, delta, sp in events:
        if prev_time is not None and t > prev_time:
            active_now = [s for s, d in active.items() if d > 0]
            if len(active_now) >= 2:
                interval_len = t - prev_time
                overlap_duration += interval_len
                overlap_events.append(
                    {
                        "start": round(prev_time, 3),
                        "end": round(t, 3),
                        "duration": round(interval_len, 3),
                        "speakers": active_now,
                    }
                )
        active[sp] = active.get(sp, 0) + delta
        prev_time = t

    return {
        "total_duration": round(total_duration, 2),
        "overlap_duration": round(overlap_duration, 2),
        "overlap_ratio": (
            round(overlap_duration / total_duration, 4) if total_duration > 0 else 0
        ),
        "n_segments": len(segments),
        "n_speakers": len(speakers),
        "overlap_events": overlap_events,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Overlap bench for a private E2E corpus"
    )
    parser.add_argument("--bench-dir", default=str(BENCH_DIR))
    parser.add_argument("--out-json", default="tmp/overlap_bench_results.json")
    args = parser.parse_args()

    bench_dir = Path(args.bench_dir)
    audio_files = sorted(bench_dir.glob("*.opus")) + sorted(bench_dir.glob("*.ogg"))

    if not audio_files:
        print(f"No .opus/.ogg files found in {bench_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(audio_files)} files in {bench_dir}")
    print(f"Service: {BASE_URL}\n")

    results = []

    for i, fpath in enumerate(audio_files, 1):
        print(f"[{i}/{len(audio_files)}] {fpath.name}")
        rec: dict = {"file": fpath.name, "status": "error", "error": None}

        try:
            # Submit (dedup: re-use if already transcribed)
            upload_resp = _upload(fpath)
            job_id = upload_resp["id"]
            deduped = upload_resp.get("deduplicated", False)

            if deduped or upload_resp.get("status") == "completed":
                print(f"  dedup hit → {job_id}")
                tr_resp = _get(f"/api/transcriptions/{job_id}")
                tr_resp.raise_for_status()
                result = tr_resp.json()
            else:
                print(f"  queued → {job_id}, polling…")
                result = _poll(job_id)

            segs = result.get("segments", [])
            stats = compute_overlap_stats(segs)
            rec.update(
                {
                    "status": "ok",
                    "job_id": job_id,
                    "deduped": deduped,
                    **stats,
                }
            )

            pct = f"{stats['overlap_ratio']*100:.1f}%"
            print(
                f"  ✓ {stats['n_segments']} segs | {stats['n_speakers']} speakers | "
                f"duration {stats['total_duration']:.0f}s | overlap {stats['overlap_duration']:.1f}s ({pct})"
            )

        except Exception as exc:
            rec["error"] = str(exc)
            print(f"  ✗ {exc}")

        results.append(rec)

    # Write JSON
    out = Path(args.out_json)
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nResults written → {out}")

    # Summary table
    ok = [r for r in results if r["status"] == "ok"]
    if ok:
        print("\n" + "=" * 72)
        print(f"{'File':<45} {'Dur':>6} {'Segs':>5} {'Spks':>5} {'Overlap':>9}")
        print("-" * 72)
        for r in ok:
            name = r["file"][:44]
            pct = f"{r['overlap_ratio']*100:.1f}%"
            print(
                f"{name:<45} {r['total_duration']:>6.0f} {r['n_segments']:>5} {r['n_speakers']:>5} {pct:>9}"
            )
        print("=" * 72)
        avg_ratio = sum(r["overlap_ratio"] for r in ok) / len(ok)
        print(
            f"  Average overlap: {avg_ratio*100:.1f}%  ({len(ok)}/{len(results)} files OK)"
        )

    errors = [r for r in results if r["status"] == "error"]
    if errors:
        print(f"\n  {len(errors)} error(s):")
        for r in errors:
            print(f"  - {r['file']}: {r['error']}")


if __name__ == "__main__":
    main()
