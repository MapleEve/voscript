"""Transcription pipeline: WhisperX (forced alignment) + pyannote + WeSpeaker ResNet34.

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
        """Lazy-load the WhisperX model (wraps faster-whisper via CT2)."""
        if self._whisper is None:
            import os
            from pathlib import Path
            import whisperx

            compute_type = "float16" if self.device == "cuda" else "int8"

            # Resolve the model path:
            #   1. If /models/faster-whisper-<size> exists (read-only mount
            #      with a pre-downloaded CT2 checkpoint tree), use it directly
            #      — zero network, zero HF hub round-trip.
            #   2. Otherwise fall back to the plain HF hub id ("large-v3" etc.)
            #      and let faster-whisper resolve via HF_HOME=/cache.
            # We NEVER pass download_root="/models" — that path is bind-mounted
            # read-only, and HF hub's snapshot layout (refs/main, blobs/…)
            # would need to write there.
            local_dir = Path("/models") / f"faster-whisper-{self.model_size}"
            model_ref = str(local_dir) if local_dir.exists() else self.model_size

            logger.info(
                "Loading WhisperX model %s on %s (compute_type=%s)",
                model_ref,
                self.device,
                compute_type,
            )
            self._whisper = whisperx.load_model(
                model_ref,
                device=self.device,
                compute_type=compute_type,
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

    def transcribe(self, audio_path: str, language: str = "zh") -> dict:
        """Run WhisperX transcription and return the raw result dict.

        Parameters
        ----------
        audio_path:
            Path to an audio file that WhisperX can load (wav, mp3, m4a, …).
        language:
            BCP-47 language code. Pass an empty string ("") to let WhisperX
            auto-detect the language from the first 30 seconds of audio
            (mirrors the old faster-whisper auto-detect behaviour). The
            container default is "zh" (Mandarin).

        Returns
        -------
        dict with at minimum:
            "segments" — list of segment dicts (start, end, text)
            "language" — detected or supplied language code
        """
        import whisperx

        audio = whisperx.load_audio(audio_path)

        # WhisperX expects language=None for auto-detect; we map "" -> None.
        lang_arg = language if language else None
        logger.info("Starting WhisperX transcription (language=%s)", lang_arg or "auto")

        result = self.whisper.transcribe(audio, language=lang_arg, batch_size=16)
        logger.info("WhisperX transcription done: language=%s", result.get("language"))
        return result

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
        self,
        transcription_result: dict,
        diarization_turns: list[dict],
        audio_path: str,
    ) -> list[dict]:
        """Force-align segments to word level, then assign speaker labels.

        Step 1 — WhisperX forced alignment (wav2vec2):
            Attempts to attach per-word timestamps to every segment. If
            alignment fails (e.g. no wav2vec2 model for the detected language,
            or a network error), we fall back gracefully to the pre-alignment
            segments without word-level timestamps and log a warning.

        Step 2 — Speaker assignment by time-overlap:
            Same logic as the previous overlap-based aligner, now operating on
            the (potentially word-enriched) aligned segments. For each segment
            we pick the pyannote turn with the greatest time overlap; midpoint
            fallback is kept for segments with no overlap.

        Parameters
        ----------
        transcription_result:
            Raw dict returned by ``transcribe()``.  Must contain a "segments"
            key and a "language" key.
        diarization_turns:
            List of ``{"start", "end", "speaker"}`` dicts from pyannote.
        audio_path:
            Path to the audio file — required by ``whisperx.load_audio`` for
            the alignment step.

        Returns
        -------
        List of segment dicts:
            ``{"start", "end", "text", "speaker"}`` always present.
            ``"words"`` key present when forced alignment succeeded:
            ``[{"word": str, "start": float, "end": float, "score": float}, ...]``
        """
        import whisperx

        segments = transcription_result.get("segments", [])
        language = transcription_result.get("language", "zh") or "zh"
        audio = whisperx.load_audio(audio_path)

        # --- Step 1: forced word-level alignment ---
        try:
            align_model, align_metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
            )
            aligned_result = whisperx.align(
                segments,
                align_model,
                align_metadata,
                audio,
                self.device,
                return_char_alignments=False,
            )
            segments = aligned_result.get("segments", segments)
            logger.info("WhisperX forced alignment succeeded for language=%s", language)
        except Exception as exc:
            logger.warning(
                "WhisperX forced alignment failed for language=%s (%s); "
                "continuing without word-level timestamps.",
                language,
                exc,
            )
            # segments stays as the pre-alignment list from transcription_result

        # --- Step 2: speaker assignment by time overlap ---
        result_segments = []
        for seg in segments:
            seg_start = seg.get("start", 0.0)
            seg_end = seg.get("end", 0.0)
            seg_mid = (seg_start + seg_end) / 2
            best_speaker = "UNKNOWN"
            best_overlap = 0.0

            for turn in diarization_turns:
                overlap_start = max(seg_start, turn["start"])
                overlap_end = min(seg_end, turn["end"])
                overlap = max(0.0, overlap_end - overlap_start)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = turn["speaker"]

            # Fallback: no time-overlap found — use segment midpoint
            if best_speaker == "UNKNOWN":
                for turn in diarization_turns:
                    if turn["start"] <= seg_mid <= turn["end"]:
                        best_speaker = turn["speaker"]
                        break

            out: dict = {
                "start": round(seg_start, 3),
                "end": round(seg_end, 3),
                "text": seg.get("text", "").strip(),
                "speaker": best_speaker,
            }

            # Include word-level timestamps when present; normalise to plain
            # Python dicts so JSON serialisation in main.py never hits numpy
            # scalars or unexpected types.
            raw_words = seg.get("words")
            if raw_words:
                out["words"] = [
                    {
                        "word": str(w.get("word", "")),
                        "start": round(float(w.get("start", 0.0)), 3),
                        "end": round(float(w.get("end", 0.0)), 3),
                        "score": round(float(w.get("score", 0.0)), 4),
                    }
                    for w in raw_words
                ]

            result_segments.append(out)

        return result_segments

    def process(
        self,
        audio_path: str,
        language: str = "zh",
        min_speakers: int = None,
        max_speakers: int = None,
    ) -> dict:
        """Full pipeline: transcribe → diarize → forced-align → extract embeddings."""
        logger.info("Starting transcription: %s", audio_path)
        transcription_result = self.transcribe(audio_path, language=language)
        logger.info(
            "Transcription done: %d segments",
            len(transcription_result.get("segments", [])),
        )

        logger.info("Starting diarization")
        turns = self.diarize(
            audio_path, min_speakers=min_speakers, max_speakers=max_speakers
        )
        logger.info("Diarization done: %d turns", len(turns))

        logger.info("Running forced alignment and assigning speakers")
        aligned = self.align_segments(transcription_result, turns, audio_path)
        logger.info("Alignment done: %d segments", len(aligned))

        logger.info("Extracting speaker embeddings")
        embeddings = self.extract_speaker_embeddings(audio_path, turns)
        logger.info("Extracted embeddings for %d speakers", len(embeddings))

        return {
            "segments": aligned,
            "speaker_embeddings": embeddings,
            "unique_speakers": list(embeddings.keys()),
        }
