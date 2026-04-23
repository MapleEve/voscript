"""Default provider for optional audio enhancement."""

from __future__ import annotations

import logging

from config import DENOISE_MODEL, DENOISE_SNR_THRESHOLD
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

        _df_model, _df_state, _ = _df_pkg.init_df()
        logger.info("DeepFilterNet model loaded")
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

        if effective_model == "deepfilternet":
            import torch
            import torchaudio

            snr_db = _estimate_snr(request.wav_path)
            if snr_db >= threshold:
                logger.info("DeepFilterNet skipped (SNR=%.1fdB, clean audio)", snr_db)
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
                "DeepFilterNet: denoised %s -> %s",
                request.wav_path.name,
                out_path.name,
            )

        elif effective_model == "noisereduce":
            import noisereduce as nr
            import soundfile as sf

            data, sr = sf.read(str(request.wav_path), dtype="float32")
            reduced = nr.reduce_noise(y=data, sr=sr, stationary=True)
            sf.write(str(out_path), reduced, sr)
            logger.info(
                "noisereduce: denoised %s -> %s",
                request.wav_path.name,
                out_path.name,
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
