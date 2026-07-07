"""Audio metadata helpers used before memory-sensitive processing."""

from __future__ import annotations

from pathlib import Path


def audio_duration_seconds(path: Path | str) -> float | None:
    """Return audio duration from metadata without loading the full waveform."""

    try:
        import torchaudio

        info = torchaudio.info(str(path))
        sample_rate = getattr(info, "sample_rate", 0) or 0
        num_frames = getattr(info, "num_frames", 0) or 0
        if sample_rate <= 0 or num_frames < 0:
            return None
        return num_frames / sample_rate
    except Exception:
        return None


__all__ = ["audio_duration_seconds"]
