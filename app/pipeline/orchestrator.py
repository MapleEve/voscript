"""Transcription pipeline: WhisperX (forced alignment) + pyannote + WeSpeaker ResNet34.

NOTE: pyannote/wespeaker-voxceleb-resnet34-LM is a gated HuggingFace model.
Users must visit https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
and click "Agree and access repository" (same process as
pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0) before the
model can be downloaded at runtime. A missing or invalid HF_TOKEN, or a token
whose owner has not accepted the gating agreement, will raise an HTTP 403 error
on the first call to extract_speaker_embeddings().
"""

import logging
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from config import DEVICE, HF_TOKEN, PYANNOTE_MIN_DURATION_OFF, WHISPER_MODEL
from infra.cuda_devices import select_best_cuda_device
from infra.huggingface_models import (
    configure_huggingface_runtime,
    resolve_hf_model_ref,
)
from providers.asr import transcribe_audio
from providers.diarization import align_diarized_segments, run_pyannote_diarization
from providers.embedding import extract_embeddings_for_turns

from .contracts import PipelineRequest
from .runner import PipelineRunner

logger = logging.getLogger(__name__)
configure_huggingface_runtime()

_TRUSTED_PYANNOTE_TASK_GLOBAL_NAMES = (
    "Problem",
    "Specifications",
    "Resolution",
)


def _trusted_pyannote_checkpoint_globals() -> list[type]:
    """Return object types trusted for pyannote checkpoint loading."""

    torch_version_type = getattr(
        getattr(torch, "torch_version", None),
        "TorchVersion",
        None,
    )
    trusted_globals = []
    if torch_version_type is not None:
        trusted_globals.append(torch_version_type)

    try:
        import pyannote.audio.core.task as pyannote_task
    except ImportError:
        pyannote_task = None
    if pyannote_task is not None:
        for name in _TRUSTED_PYANNOTE_TASK_GLOBAL_NAMES:
            trusted_type = getattr(pyannote_task, name, None)
            if trusted_type is not None:
                trusted_globals.append(trusted_type)

    return trusted_globals


def _trusted_pyannote_checkpoint_context():
    """Scope the trusted object allowlist to pyannote checkpoint loads only."""

    serialization = getattr(torch, "serialization", None)
    safe_globals = getattr(serialization, "safe_globals", None)
    trusted_globals = _trusted_pyannote_checkpoint_globals()
    if safe_globals is None or not trusted_globals:
        return nullcontext()
    return safe_globals(trusted_globals)


def _load_trusted_pyannote_model(
    from_pretrained,
    model_ref: str,
    hub_auth: str | None,
):
    auth_kwargs = {"use_auth_" + "token": hub_auth}
    with _trusted_pyannote_checkpoint_context():
        return from_pretrained(model_ref, **auth_kwargs)


def _is_local_model_ref(model_ref: str | Path) -> bool:
    if isinstance(model_ref, Path):
        return True
    path = Path(model_ref).expanduser()
    return path.is_absolute() or model_ref.startswith((".", "~"))


def _resolve_local_pyannote_file(model_ref: str | Path, snapshot_filename: str) -> str:
    """Convert a local HF snapshot directory into pyannote's expected file path."""

    if not _is_local_model_ref(model_ref):
        return str(model_ref)

    local_path = Path(model_ref).expanduser()
    if local_path.is_dir():
        local_path = local_path / snapshot_filename

    if not local_path.is_file():
        raise FileNotFoundError(f"Local pyannote model file not found: {local_path}")

    return str(local_path)


def _faster_whisper_device_kwargs(device: str) -> dict[str, Any]:
    """Translate torch-style CUDA device strings to faster-whisper kwargs."""

    if not device.startswith("cuda:"):
        return {"device": device}

    device_kind, _, raw_index = device.partition(":")
    if raw_index.isdigit():
        return {"device": device_kind, "device_index": int(raw_index)}
    return {"device": device}


class TranscriptionPipeline:
    def __init__(
        self,
        model_size: str = None,
        device: str = None,
        hf_token: str = None,
    ):
        self._configured_device = device or DEVICE
        self.device = self._configured_device
        self.model_size = model_size or WHISPER_MODEL
        self.hf_token = hf_token or HF_TOKEN
        self._whisper = None
        self._diarization = None
        self._embedding_model = None
        self._runner = None

    @property
    def runner(self) -> PipelineRunner:
        runner = getattr(self, "_runner", None)
        if runner is None:
            runner = PipelineRunner()
            self._runner = runner
        return runner

    def has_loaded_models(self) -> bool:
        return any(
            getattr(self, name, None) is not None
            for name in ("_whisper", "_diarization", "_embedding_model")
        )

    def unload_models(self) -> None:
        self._whisper = None
        self._diarization = None
        self._embedding_model = None

    def _select_device_for_lazy_load(self) -> None:
        configured_device = getattr(self, "_configured_device", self.device)
        if self.has_loaded_models() or not configured_device.startswith("cuda"):
            return
        self.device = select_best_cuda_device(configured_device)

    @property
    def whisper(self):
        """Lazy-load faster-whisper directly.

        We deliberately do NOT use ``whisperx.load_model`` here: keeping ASR on
        faster-whisper directly avoids WhisperX wrapper compatibility issues
        around ``faster_whisper.TranscriptionOptions``. WhisperX is used only
        for forced alignment below (``whisperx.align``), which is decoupled
        from the transcriber.
        """
        if self._whisper is None:
            self._select_device_for_lazy_load()
            # faster_whisper 按需 lazy import，避免在不使用 whisper 的进程里加载 GPU 库
            from faster_whisper import WhisperModel

            compute_type = "float16" if self.device.startswith("cuda") else "int8"
            local_dir = Path("/models") / f"faster-whisper-{self.model_size}"
            model_ref = str(local_dir) if local_dir.exists() else self.model_size
            logger.info(
                "Loading faster-whisper %s on %s (compute_type=%s)",
                model_ref,
                self.device,
                compute_type,
            )
            self._whisper = WhisperModel(
                model_ref,
                **_faster_whisper_device_kwargs(self.device),
                compute_type=compute_type,
            )
        return self._whisper

    @property
    def diarization(self):
        if self._diarization is None:
            self._select_device_for_lazy_load()
            from pyannote.audio import Pipeline as PyannotePipeline

            model_ref = resolve_hf_model_ref(
                "pyannote/speaker-diarization-3.1",
                token=self.hf_token,
                purpose="pyannote diarization",
            )
            model_ref = _resolve_local_pyannote_file(model_ref, "config.yaml")
            logger.info("Loading pyannote diarization model")
            self._diarization = _load_trusted_pyannote_model(
                PyannotePipeline.from_pretrained,
                model_ref,
                self.hf_token,
            )
            _dev = self.device if ":" in self.device else "cuda:0"
            if self.device.startswith("cuda"):
                self._diarization.to(torch.device(_dev))
            # Suppress over-segmentation of short backchannel turns
            try:
                if hasattr(self._diarization, "_binarize") and hasattr(
                    self._diarization._binarize, "min_duration_off"
                ):
                    self._diarization._binarize.min_duration_off = (
                        PYANNOTE_MIN_DURATION_OFF
                    )
                    logger.info(
                        "Set diarization min_duration_off=%.2f",
                        PYANNOTE_MIN_DURATION_OFF,
                    )
            except Exception as exc:
                logger.warning("Could not set min_duration_off: %s", exc)
        return self._diarization

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            self._select_device_for_lazy_load()
            from pyannote.audio import Inference, Model

            model_ref = resolve_hf_model_ref(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                token=self.hf_token,
                purpose="WeSpeaker speaker encoder",
            )
            model_ref = _resolve_local_pyannote_file(model_ref, "pytorch_model.bin")
            logger.info("Loading WeSpeaker speaker encoder")
            model = _load_trusted_pyannote_model(
                Model.from_pretrained,
                model_ref,
                self.hf_token,
            )
            model = model.to(torch.device(self.device))
            # window="whole" returns one embedding vector per full chunk —
            # exactly what we need for per-turn embeddings.
            self._embedding_model = Inference(model, window="whole")
        return self._embedding_model

    def transcribe(
        self, audio_path: str, language: str = None, no_repeat_ngram_size: int = None
    ) -> dict:
        """Compatibility entrypoint for direct ASR calls."""

        return transcribe_audio(
            self,
            audio_path,
            language=language,
            no_repeat_ngram_size=no_repeat_ngram_size,
        ).transcription_result

    def diarize(
        self, audio_path: str, min_speakers: int = None, max_speakers: int = None
    ) -> list[dict]:
        """Compatibility entrypoint for direct diarization calls."""

        return run_pyannote_diarization(
            self,
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

    def extract_speaker_embeddings(
        self, audio_path: str, turns: list[dict]
    ) -> dict[str, Any]:
        """Compatibility entrypoint for direct speaker embedding calls."""

        return extract_embeddings_for_turns(self, audio_path, turns)

    def align_segments(
        self,
        transcription_result: dict,
        diarization_turns: list[dict],
        audio_path: str,
    ) -> list[dict]:
        """Compatibility entrypoint for direct alignment calls."""

        return align_diarized_segments(
            self,
            transcription_result,
            diarization_turns,
            audio_path,
        )

    def process(
        self,
        audio_path: str,
        raw_audio_path: str = None,
        language: str = None,
        min_speakers: int = None,
        max_speakers: int = None,
        no_repeat_ngram_size: int = None,
        voiceprint_db: Any = None,
        voiceprint_threshold: float = None,
        denoise_model: str | None = None,
        snr_threshold: float | None = None,
        artifact_dir: Path | None = None,
        status_callback: Any = None,
        provider_selection: dict[str, str] | None = None,
    ) -> dict:
        """Run the stable pipeline stage order through the current implementation.

        audio_path      — upload or pre-normalized audio handed to the pipeline.
        raw_audio_path  — optional caller-managed embedding source override.
                          Falls back to the normalized audio when omitted.
        """
        request = PipelineRequest(
            audio_path=audio_path,
            raw_audio_path=raw_audio_path,
            language=language,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            no_repeat_ngram_size=no_repeat_ngram_size,
            voiceprint_db=voiceprint_db,
            voiceprint_threshold=voiceprint_threshold,
            denoise_model=denoise_model,
            snr_threshold=snr_threshold,
            artifact_dir=artifact_dir,
            status_callback=status_callback,
            provider_selection=provider_selection or {},
        )
        return self.runner.run(self, request)
