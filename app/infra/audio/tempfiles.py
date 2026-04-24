"""Cleanup helpers for generated intermediate audio files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def cleanup_generated_files(paths: Iterable[Path | str]) -> None:
    """Delete generated intermediates in reverse order, best-effort only."""

    seen: set[Path] = set()
    for raw_path in reversed(list(paths)):
        path = Path(raw_path)
        if path in seen:
            continue
        seen.add(path)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = ["cleanup_generated_files"]
