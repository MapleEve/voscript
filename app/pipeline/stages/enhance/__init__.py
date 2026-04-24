"""Stable slot for signal enhancement and denoising."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from config import DENOISE_MODEL
from providers.enhance import enhance_audio

if TYPE_CHECKING:
    from pipeline.contracts import PipelineContext


def run(context: "PipelineContext") -> None:
    """Apply the configured enhancement provider to the working WAV."""

    effective_model = (context.request.denoise_model or DENOISE_MODEL).strip().lower()
    if context.request.status_callback is not None and effective_model != "none":
        context.request.status_callback("denoising")

    input_path = Path(context.working_audio_path or context.request.audio_path)
    result = enhance_audio(
        input_path,
        model=context.request.denoise_model,
        snr_threshold=context.request.snr_threshold,
        provider_name=context.request.provider_for("enhance"),
    )

    context.working_audio_path = str(result.output_path)
    if result.applied and result.output_path != result.input_path:
        context.temporary_paths.append(result.output_path)

    context.metadata["enhance"] = {
        "status": "applied" if result.applied else "skipped",
        "model": result.model,
        "working_audio_path": context.working_audio_path,
    }
