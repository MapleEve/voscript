"""Default provider for diarization and forced-alignment handoff."""

from __future__ import annotations

import logging
import os
import re
import time
from contextlib import contextmanager
from collections.abc import Callable
from inspect import Parameter, signature
from typing import Any

from config import (
    WHISPERX_ALIGN_CACHE_ONLY,
    WHISPERX_ALIGN_DEVICE,
    WHISPERX_ALIGN_DISABLED_LANGUAGES,
    WHISPERX_ALIGN_MAX_AUDIO_DURATION_SEC,
    WHISPERX_ALIGN_MODEL_DIR,
    WHISPERX_ALIGN_MODEL_MAP,
)
from infra.audio import audio_duration_seconds
from pipeline.contracts import (
    DiarizationProvider,
    DiarizationRequest,
    DiarizationResult,
)
from postprocess.alignment import (
    build_aligned_segments,
    dedup_short_segments,
)

logger = logging.getLogger(__name__)

WHISPERX_DEFAULT_ALIGN_MODELS = {
    "zh": "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn",
}

_SAFE_MODEL_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$"
)

_GENERIC_ALIGNMENT_FAILURE_HINT = (
    "Check WHISPERX_ALIGN_MODEL_MAP, WHISPERX_ALIGN_MODEL_DIR, "
    "WHISPERX_ALIGN_CACHE_ONLY, network access, and model compatibility."
)
_TORCH_VERSION_BLOCKED_HINT = (
    "Chinese word-level alignment with the default pytorch_model.bin weights "
    "requires torch>=2.6 under recent transformers safety checks, or a trusted "
    "replacement alignment model that provides safetensors."
)


def _normalise_language(language: object) -> str:
    value = str(language or "zh").strip().lower()
    return value or "zh"


def _supports_keyword(fn: Callable[..., object], name: str) -> bool:
    try:
        parameters = signature(fn).parameters.values()
    except (TypeError, ValueError):
        return True
    return any(
        parameter.kind is Parameter.VAR_KEYWORD or parameter.name == name
        for parameter in parameters
    )


def _load_align_model_kwargs(
    load_align_model, language: str, device: str
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "language_code": language,
        "device": device,
    }
    model_name = WHISPERX_ALIGN_MODEL_MAP.get(language)
    if model_name and _supports_keyword(load_align_model, "model_name"):
        kwargs["model_name"] = model_name
    if WHISPERX_ALIGN_MODEL_DIR and _supports_keyword(load_align_model, "model_dir"):
        kwargs["model_dir"] = WHISPERX_ALIGN_MODEL_DIR
    if WHISPERX_ALIGN_CACHE_ONLY and _supports_keyword(
        load_align_model,
        "model_cache_only",
    ):
        kwargs["model_cache_only"] = True
    return kwargs


def _alignment_model_name(language: str) -> str | None:
    return WHISPERX_ALIGN_MODEL_MAP.get(language) or WHISPERX_DEFAULT_ALIGN_MODELS.get(
        language
    )


def _safe_model_metadata(model_name: str | None) -> str | None:
    if model_name is None:
        return None
    if model_name in WHISPERX_DEFAULT_ALIGN_MODELS.values():
        return model_name
    if _SAFE_MODEL_ID_RE.match(model_name):
        return model_name
    return "<custom>"


def _alignment_disabled(language: str) -> bool:
    return (
        language in WHISPERX_ALIGN_DISABLED_LANGUAGES
        and language not in WHISPERX_ALIGN_MODEL_MAP
    )


def _alignment_duration_budget_metadata(
    audio_path: str,
    *,
    language: str,
    model_metadata: str | None,
) -> dict[str, Any] | None:
    duration_s = audio_duration_seconds(audio_path)
    if (
        WHISPERX_ALIGN_MAX_AUDIO_DURATION_SEC > 0
        and duration_s is not None
        and duration_s > WHISPERX_ALIGN_MAX_AUDIO_DURATION_SEC
    ):
        return {
            "status": "skipped",
            "language": language,
            "model": model_metadata,
            "reason": "duration_budget_exceeded",
            "duration_s": round(duration_s, 3),
            "max_duration_s": WHISPERX_ALIGN_MAX_AUDIO_DURATION_SEC,
            "actionable_hint": (
                "Increase WHISPERX_ALIGN_MAX_AUDIO_DURATION_SEC only after "
                "validating memory headroom for forced alignment."
            ),
        }
    return None


def _resolve_alignment_device(pipeline) -> str:
    configured = (WHISPERX_ALIGN_DEVICE or "cpu").strip().lower()
    if configured in {"pipeline", "asr"}:
        return str(getattr(pipeline, "device", "cpu") or "cpu")
    if configured == "auto":
        selector = getattr(pipeline, "_select_device_for_lazy_load", None)
        if callable(selector):
            return str(selector("_alignment_device"))
        return str(getattr(pipeline, "device", "cpu") or "cpu")
    return configured or "cpu"


def _alignment_cache_key(
    *,
    language: str,
    model_name: str | None,
    model_source: str,
    device: str,
) -> tuple[str, str | None, str, str | None, bool, str]:
    return (
        language,
        model_name,
        model_source,
        WHISPERX_ALIGN_MODEL_DIR,
        WHISPERX_ALIGN_CACHE_ONLY,
        device,
    )


def _language_disabled_hint(language: str) -> str:
    return (
        f"Remove {language} from WHISPERX_ALIGN_DISABLED_LANGUAGES to retry "
        f"alignment, or set WHISPERX_ALIGN_MODEL_MAP={language}=<model> for a "
        "replacement model."
    )


def _classify_alignment_failure(exc: Exception) -> tuple[str, str]:
    message = str(exc).lower()
    if "torch.load" in message and "v2.6" in message and "safetensors" in message:
        return "torch_version_blocked", _TORCH_VERSION_BLOCKED_HINT
    return "load_or_align_failed", _GENERIC_ALIGNMENT_FAILURE_HINT


@contextmanager
def _cache_only_alignment_environment():
    if not WHISPERX_ALIGN_CACHE_ONLY:
        yield
        return

    previous = {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
    }
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _parse_torch_version(version: str) -> tuple[int, ...]:
    parts = []
    for part in version.split("+", 1)[0].split("."):
        digits = "".join(char for char in part if char.isdigit())
        if digits == "":
            break
        parts.append(int(digits))
    return tuple(parts)


def _torch_preflight_message(language: str, model_name: str | None) -> str | None:
    if language != "zh" or model_name != WHISPERX_DEFAULT_ALIGN_MODELS["zh"]:
        return None
    try:
        import torch
    except Exception:
        return None

    version = _parse_torch_version(getattr(torch, "__version__", ""))
    if version and version < (2, 6):
        return (
            "WhisperX zh alignment preflight: default model uses PyTorch weights; "
            "transformers may block loading with torch<2.6 unless safetensors are "
            "available."
        )
    return None


def run_pyannote_diarization(
    pipeline,
    audio_path: str,
    *,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> list[dict[str, object]]:
    """Run the current pyannote diarization resource and return speaker turns."""

    kwargs = {}
    if min_speakers:
        kwargs["min_speakers"] = min_speakers
    if max_speakers:
        kwargs["max_speakers"] = max_speakers

    diarization_model = pipeline.diarization
    processing_started = time.perf_counter()
    result = diarization_model(audio_path, **kwargs)
    turns: list[dict[str, object]] = []
    for turn, _, speaker in result.itertracks(yield_label=True):
        turns.append(
            {
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker,
            }
        )
    elapsed_s = time.perf_counter() - processing_started
    logger.info(
        "diarization_processing_timing model=pyannote elapsed_s=%.3f device=%s turn_count=%d speaker_count=%d",
        elapsed_s,
        getattr(pipeline, "_diarization_device", None)
        or getattr(pipeline, "device", ""),
        len(turns),
        len({turn["speaker"] for turn in turns}),
    )
    return turns


def align_diarized_segments_with_metadata(
    pipeline,
    transcription_result: dict[str, object],
    diarization_turns: list[dict[str, object]],
    audio_path: str,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    """Align ASR output and attach diarization speaker labels."""

    segments = transcription_result.get("segments", [])
    language = _normalise_language(transcription_result.get("language"))
    model_source = (
        "override" if language in WHISPERX_ALIGN_MODEL_MAP else "whisperx_default"
    )
    model_name = _alignment_model_name(language)
    model_metadata = _safe_model_metadata(model_name)

    if _alignment_disabled(language):
        metadata = {
            "status": "skipped",
            "language": language,
            "model": model_metadata,
            "reason": "language_disabled",
            "actionable_hint": _language_disabled_hint(language),
        }
        logger.info(
            "WhisperX forced alignment skipped for language=%s reason=%s",
            language,
            metadata["reason"],
        )
        return build_aligned_segments(segments, diarization_turns), metadata

    budget_metadata = _alignment_duration_budget_metadata(
        audio_path,
        language=language,
        model_metadata=model_metadata,
    )
    if budget_metadata is not None:
        logger.warning(
            "WhisperX forced alignment skipped for language=%s reason=%s "
            "duration_s=%.3f max_duration_s=%.3f",
            language,
            budget_metadata["reason"],
            budget_metadata["duration_s"],
            budget_metadata["max_duration_s"],
        )
        return build_aligned_segments(segments, diarization_turns), budget_metadata

    try:
        import whisperx

        preflight_message = _torch_preflight_message(language, model_name)
        if preflight_message:
            logger.info(preflight_message)
        alignment_device = _resolve_alignment_device(pipeline)
        audio = whisperx.load_audio(audio_path)
        load_kwargs = _load_align_model_kwargs(
            whisperx.load_align_model,
            language,
            alignment_device,
        )
        cache_key = _alignment_cache_key(
            language=language,
            model_name=model_name,
            model_source=model_source,
            device=alignment_device,
        )
        cached_key = getattr(pipeline, "_alignment_cache_key", None)
        align_model = getattr(pipeline, "_alignment_model", None)
        align_metadata = getattr(pipeline, "_alignment_metadata", None)
        if (
            cached_key == cache_key
            and align_model is not None
            and align_metadata is not None
        ):
            logger.info(
                "Reusing WhisperX alignment model (hot reuse, language=%s, model_source=%s, device=%s)",
                language,
                model_source,
                alignment_device,
            )
        else:
            setattr(pipeline, "_alignment_model", None)
            setattr(pipeline, "_alignment_metadata", None)
            setattr(pipeline, "_alignment_cache_key", None)
            load_started = time.perf_counter()
            with _cache_only_alignment_environment():
                align_model, align_metadata = whisperx.load_align_model(
                    **load_kwargs,
                )
            setattr(pipeline, "_alignment_model", align_model)
            setattr(pipeline, "_alignment_metadata", align_metadata)
            setattr(pipeline, "_alignment_cache_key", cache_key)
            setattr(pipeline, "_alignment_device", alignment_device)
            logger.info(
                "Loaded WhisperX alignment model in %.2fs "
                "(cold_load=True, language=%s, model_source=%s, device=%s)",
                time.perf_counter() - load_started,
                language,
                model_source,
                alignment_device,
            )
        processing_started = time.perf_counter()
        aligned_result = whisperx.align(
            segments,
            align_model,
            align_metadata,
            audio,
            alignment_device,
            return_char_alignments=False,
        )
        processing_elapsed_s = time.perf_counter() - processing_started
        segments = aligned_result.get("segments", segments)
        logger.info(
            "alignment_processing_timing model=whisperx elapsed_s=%.3f language=%s segment_count=%d device=%s",
            processing_elapsed_s,
            language,
            len(segments),
            alignment_device,
        )
        logger.info("WhisperX forced alignment succeeded for language=%s", language)
        metadata = {
            "status": "succeeded",
            "language": language,
            "model": model_metadata,
            "model_source": model_source,
            "cache_only": WHISPERX_ALIGN_CACHE_ONLY,
        }
    except Exception as exc:
        error_type = exc.__class__.__name__
        reason, actionable_hint = _classify_alignment_failure(exc)
        metadata = {
            "status": "failed",
            "language": language,
            "model": model_metadata,
            "reason": reason,
            "error_type": error_type,
            "model_source": model_source,
            "cache_only": WHISPERX_ALIGN_CACHE_ONLY,
            "actionable_hint": actionable_hint,
        }
        logger.warning(
            "WhisperX forced alignment failed for language=%s model_source=%s "
            "reason=%s error_type=%s; "
            "continuing without word-level timestamps.",
            language,
            model_source,
            reason,
            error_type,
        )

    return build_aligned_segments(segments, diarization_turns), metadata


def align_diarized_segments(
    pipeline,
    transcription_result: dict[str, object],
    diarization_turns: list[dict[str, object]],
    audio_path: str,
) -> list[dict[str, object]]:
    """Compatibility wrapper that returns aligned segments only."""

    aligned_segments, _metadata = align_diarized_segments_with_metadata(
        pipeline,
        transcription_result,
        diarization_turns,
        audio_path,
    )
    return aligned_segments


class PipelineMethodDiarizationProvider(DiarizationProvider):
    """Run diarization and alignment through pipeline-owned resources."""

    def diarize(self, request: DiarizationRequest) -> DiarizationResult:
        turns = run_pyannote_diarization(
            request.pipeline,
            request.audio_path,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers,
        )
        aligned, alignment_metadata = align_diarized_segments_with_metadata(
            request.pipeline,
            request.transcription_result,
            turns,
            request.audio_path,
        )
        deduped = dedup_short_segments(aligned)
        return DiarizationResult(
            turns=turns,
            aligned_segments=deduped,
            dedup_removed=len(aligned) - len(deduped),
            metadata={"alignment": alignment_metadata},
        )


default_diarization_provider = PipelineMethodDiarizationProvider()


__all__ = [
    "PipelineMethodDiarizationProvider",
    "align_diarized_segments",
    "align_diarized_segments_with_metadata",
    "default_diarization_provider",
    "run_pyannote_diarization",
]
