"""Default provider for optional audio enhancement."""

from __future__ import annotations

import logging
import time

from config import (
    DENOISE_MAX_AUDIO_DURATION_SEC,
    DENOISE_MODEL,
    DENOISE_SNR_THRESHOLD,
)
from infra.audio import audio_duration_seconds
from pipeline.contracts import (
    AudioEnhancementProvider,
    AudioEnhancementRequest,
    AudioEnhancementResult,
)

logger = logging.getLogger(__name__)

_df_model = None
_df_state = None


def _load_deepfilternet():
    global _df_model, _df_state
    if _df_model is None:
        import df as _df_pkg

        load_started = time.perf_counter()
        _df_model, _df_state, _ = _df_pkg.init_df()
        logger.info(
            "Loaded DeepFilterNet model in %.2fs (cold_load=True)",
            time.perf_counter() - load_started,
        )
    else:
        logger.info("Reusing DeepFilterNet model (hot reuse)")
    return _df_model, _df_state


def _estimate_snr(wav_path):
    """Estimate signal-to-noise ratio (dB) using an energy-based heuristic."""
    import math

    import torchaudio

    waveform, sr = torchaudio.load(str(wav_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)

    frame_len = max(1, int(sr * 0.03))
    num_frames = len(waveform) // frame_len
    if num_frames < 5:
        return float("inf")

    frames = waveform[: num_frames * frame_len].reshape(num_frames, frame_len)
    frame_rms = frames.pow(2).mean(dim=1).sqrt()

    sorted_rms, _ = frame_rms.sort()
    noise_cutoff = max(1, int(num_frames * 0.20))
    noise_rms = sorted_rms[:noise_cutoff].mean().item()
    speech_rms = sorted_rms[noise_cutoff:].mean().item()

    if noise_rms < 1e-9:
        return float("inf")

    return 10.0 * math.log10((speech_rms / noise_rms) ** 2)


def _duration_exceeds_denoise_budget(wav_path) -> tuple[bool, float | None]:
    duration_s = audio_duration_seconds(wav_path)
    if (
        DENOISE_MAX_AUDIO_DURATION_SEC > 0
        and duration_s is not None
        and duration_s > DENOISE_MAX_AUDIO_DURATION_SEC
    ):
        return True, duration_s
    return False, duration_s


class ConditionalDenoiseEnhancer(AudioEnhancementProvider):
    """Apply denoising only when configured and warranted by the signal."""

    def enhance(self, request: AudioEnhancementRequest) -> AudioEnhancementResult:
        effective_model = (request.model or DENOISE_MODEL).strip().lower()
        if effective_model == "none":
            return AudioEnhancementResult(
                input_path=request.wav_path,
                output_path=request.wav_path,
                applied=False,
                model=effective_model,
            )

        threshold = (
            request.snr_threshold
            if request.snr_threshold is not None
            else DENOISE_SNR_THRESHOLD
        )
        out_path = request.wav_path.with_suffix(".denoised.wav")

        if effective_model in {"deepfilternet", "noisereduce"}:
            over_budget, duration_s = _duration_exceeds_denoise_budget(request.wav_path)
            if over_budget:
                logger.warning(
                    "Denoise skipped by duration budget "
                    "model=%s duration_s=%.3f max_duration_s=%.3f",
                    effective_model,
                    duration_s,
                    DENOISE_MAX_AUDIO_DURATION_SEC,
                )
                return AudioEnhancementResult(
                    input_path=request.wav_path,
                    output_path=request.wav_path,
                    applied=False,
                    model=effective_model,
                )

        if effective_model == "deepfilternet":
            import torch
            import torchaudio

            processing_started = time.perf_counter()
            snr_db = _estimate_snr(request.wav_path)
            if snr_db >= threshold:
                logger.info("DeepFilterNet skipped (SNR=%.1fdB, clean audio)", snr_db)
                logger.info(
                    "enhance_processing_timing model=deepfilternet elapsed_s=%.3f "
                    "applied=False reason=clean_audio snr_db=%.1f threshold=%.1f",
                    time.perf_counter() - processing_started,
                    snr_db,
                    threshold,
                )
                return AudioEnhancementResult(
                    input_path=request.wav_path,
                    output_path=request.wav_path,
                    applied=False,
                    model=effective_model,
                )

            logger.info(
                "DeepFilterNet applying (SNR=%.1fdB < %.1fdB threshold)",
                snr_db,
                threshold,
            )
            df_model, df_state = _load_deepfilternet()
            import df as _df_pkg

            audio, sr = torchaudio.load(str(request.wav_path))
            input_sample_rate = sr
            if sr != df_state.sr():
                audio = torchaudio.functional.resample(audio, sr, df_state.sr())
            audio = audio.contiguous()
            with torch.backends.cudnn.flags(enabled=False):
                enhanced = _df_pkg.enhance(df_model, df_state, audio)
            torchaudio.save(
                str(out_path),
                enhanced.unsqueeze(0) if enhanced.dim() == 1 else enhanced,
                df_state.sr(),
            )
            logger.info(
                "enhance_processing_timing model=deepfilternet elapsed_s=%.3f "
                "applied=True reason=enhanced snr_db=%.1f threshold=%.1f "
                "device=%s input_sample_rate=%d output_sample_rate=%d",
                time.perf_counter() - processing_started,
                snr_db,
                threshold,
                getattr(audio, "device", "unknown"),
                input_sample_rate,
                df_state.sr(),
            )

        elif effective_model == "noisereduce":
            import noisereduce as nr
            import soundfile as sf

            processing_started = time.perf_counter()
            data, sr = sf.read(str(request.wav_path), dtype="float32")
            reduced = nr.reduce_noise(y=data, sr=sr, stationary=True)
            sf.write(str(out_path), reduced, sr)
            logger.info(
                "enhance_processing_timing model=noisereduce elapsed_s=%.3f "
                "applied=True reason=enhanced sample_rate=%d",
                time.perf_counter() - processing_started,
                sr,
            )

        else:
            logger.warning(
                "Unknown DENOISE_MODEL=%r - skipping denoising",
                effective_model,
            )
            return AudioEnhancementResult(
                input_path=request.wav_path,
                output_path=request.wav_path,
                applied=False,
                model=effective_model,
            )

        return AudioEnhancementResult(
            input_path=request.wav_path,
            output_path=out_path,
            applied=True,
            model=effective_model,
        )


default_enhance_provider = ConditionalDenoiseEnhancer()
default_audio_enhancer = default_enhance_provider


__all__ = [
    "ConditionalDenoiseEnhancer",
    "default_audio_enhancer",
    "default_enhance_provider",
]
