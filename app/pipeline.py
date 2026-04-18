"""Transcription pipeline: faster-whisper + pyannote + WeSpeaker ResNet34.

NOTE: pyannote/wespeaker-voxceleb-resnet34-LM is a gated HuggingFace model.
Users must visit https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
and click "Agree and access repository" (same process as
pyannote/speaker-diarization-3.1 and pyannote/segmentation-3.0) before the
model can be downloaded at runtime. A missing or invalid HF_TOKEN, or a token
whose owner has not accepted the gating agreement, will raise an HTTP 403 error
on the first call to extract_speaker_embeddings().
"""

import os
import logging
import numpy as np
import torch
import torchaudio
from pathlib import Path

logger = logging.getLogger(__name__)


class TranscriptionPipeline:
    def __init__(
        self,
        model_size: str = None,
        device: str = None,
        hf_token: str = None,
    ):
        self.device = device or os.getenv("DEVICE", "cuda")
        self.model_size = model_size or os.getenv("WHISPER_MODEL", "large-v3")
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self._whisper = None
        self._diarization = None
        self._embedding_model = None

    @property
    def whisper(self):
        if self._whisper is None:
            from faster_whisper import WhisperModel

            # Try local model path first, fall back to HF download
            local_model = f"/models/faster-whisper-{self.model_size}"
            model_path = local_model if Path(local_model).exists() else self.model_size
            logger.info(
                "Loading faster-whisper %s on %s (path: %s)",
                self.model_size,
                self.device,
                model_path,
            )
            compute = "float16" if self.device == "cuda" else "int8"
            self._whisper = WhisperModel(
                model_path, device=self.device, compute_type=compute
            )
        return self._whisper

    @property
    def diarization(self):
        if self._diarization is None:
            from pyannote.audio import Pipeline as PyannotePipeline

            logger.info("Loading pyannote speaker-diarization-3.1")
            self._diarization = PyannotePipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=self.hf_token,
            )
            if self.device == "cuda":
                self._diarization.to(torch.device("cuda"))
        return self._diarization

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            from pyannote.audio import Model, Inference

            logger.info("Loading WeSpeaker ResNet34 speaker encoder")
            model = Model.from_pretrained(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                use_auth_token=self.hf_token,
            )
            model = model.to(torch.device(self.device))
            # window="whole" returns one embedding vector per full chunk —
            # exactly what we need for per-turn embeddings.
            self._embedding_model = Inference(model, window="whole")
        return self._embedding_model

    def transcribe(self, audio_path: str, language: str = "zh") -> list[dict]:
        """Run faster-whisper and return segments with timestamps."""
        segments_iter, info = self.whisper.transcribe(
            audio_path,
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        logger.info(
            "Detected language: %s (prob %.2f)",
            info.language,
            info.language_probability,
        )
        segments = []
        for seg in segments_iter:
            segments.append(
                {
                    "start": round(seg.start, 3),
                    "end": round(seg.end, 3),
                    "text": seg.text.strip(),
                }
            )
        return segments

    def diarize(
        self, audio_path: str, min_speakers: int = None, max_speakers: int = None
    ) -> list[dict]:
        """Run pyannote diarization and return speaker turns."""
        kwargs = {}
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers
        result = self.diarization(audio_path, **kwargs)
        turns = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            turns.append(
                {
                    "start": round(turn.start, 3),
                    "end": round(turn.end, 3),
                    "speaker": speaker,
                }
            )
        return turns

    def extract_speaker_embeddings(
        self, audio_path: str, turns: list[dict]
    ) -> dict[str, np.ndarray]:
        """Extract one averaged WeSpeaker ResNet34 embedding per speaker.

        Output: {speaker_label: np.ndarray of shape (embedding_dim,)}
        WeSpeaker ResNet34 produces ~256-dim embeddings (vs ECAPA-TDNN 192-dim).
        The downstream VoiceprintDB is dim-agnostic and infers the dimension on
        first insert, so no other changes are required.
        """
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
            sr = 16000
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        speaker_segments: dict[str, list] = {}
        for t in turns:
            spk = t["speaker"]
            start_sample = int(t["start"] * sr)
            end_sample = int(t["end"] * sr)
            chunk = waveform[:, start_sample:end_sample]
            if chunk.shape[1] < sr:  # skip segments shorter than 1s
                continue
            speaker_segments.setdefault(spk, []).append(chunk)

        embeddings = {}
        for spk, chunks in speaker_segments.items():
            emb_list = []
            # Use up to 10 longest segments for embedding
            chunks.sort(key=lambda c: c.shape[1], reverse=True)
            for chunk in chunks[:10]:
                # Inference.__call__ accepts a dict with waveform (1, T) tensor
                # and sample_rate; window="whole" returns one ndarray per chunk.
                emb = self.embedding_model(
                    {"waveform": chunk.to(self.device), "sample_rate": 16000}
                )
                emb_list.append(np.asarray(emb))
            if emb_list:
                embeddings[spk] = np.mean(emb_list, axis=0)
        return embeddings

    def align_segments(
        self, whisper_segments: list[dict], diarization_turns: list[dict]
    ) -> list[dict]:
        """Assign a speaker label to each whisper segment by time overlap."""
        aligned = []
        for seg in whisper_segments:
            seg_mid = (seg["start"] + seg["end"]) / 2
            best_speaker = "UNKNOWN"
            best_overlap = 0

            for turn in diarization_turns:
                overlap_start = max(seg["start"], turn["start"])
                overlap_end = min(seg["end"], turn["end"])
                overlap = max(0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn["speaker"]

            # Fallback: if no overlap, use midpoint
            if best_speaker == "UNKNOWN":
                for turn in diarization_turns:
                    if turn["start"] <= seg_mid <= turn["end"]:
                        best_speaker = turn["speaker"]
                        break

            aligned.append({**seg, "speaker": best_speaker})
        return aligned

    def process(
        self,
        audio_path: str,
        language: str = "zh",
        min_speakers: int = None,
        max_speakers: int = None,
    ) -> dict:
        """Full pipeline: transcribe → diarize → align → extract embeddings."""
        logger.info("Starting transcription: %s", audio_path)
        whisper_segments = self.transcribe(audio_path, language=language)
        logger.info("Transcription done: %d segments", len(whisper_segments))

        logger.info("Starting diarization")
        turns = self.diarize(
            audio_path, min_speakers=min_speakers, max_speakers=max_speakers
        )
        logger.info("Diarization done: %d turns", len(turns))

        logger.info("Aligning segments with speakers")
        aligned = self.align_segments(whisper_segments, turns)

        logger.info("Extracting speaker embeddings")
        embeddings = self.extract_speaker_embeddings(audio_path, turns)
        logger.info("Extracted embeddings for %d speakers", len(embeddings))

        return {
            "segments": aligned,
            "speaker_embeddings": embeddings,
            "unique_speakers": list(embeddings.keys()),
        }
