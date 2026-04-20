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
        self._osd = None
        self._osd_onset = 0.5
        self._clearvoice = None

    @property
    def whisper(self):
        """Lazy-load faster-whisper directly.

        We deliberately do NOT use ``whisperx.load_model`` here: whisperx 3.1.x
        (the only line compatible with our ``torch==2.4.1`` + ``pyannote==3.1.1``
        pins) was built against an older ``faster_whisper.TranscriptionOptions``
        schema and crashes with newer faster-whisper versions.  whisperx is
        used only for forced alignment below (``whisperx.align``), which is
        decoupled from the transcriber.
        """
        if self._whisper is None:
            from pathlib import Path
            from faster_whisper import WhisperModel

            compute_type = "float16" if self.device == "cuda" else "int8"
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
            _dev = self.device if ":" in self.device else "cuda:0"
            if self.device.startswith("cuda"):
                self._diarization.to(torch.device(_dev))
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

    @property
    def osd_pipeline(self):
        if self._osd is None:
            from pyannote.audio import Model
            from pyannote.audio.pipelines import OverlappedSpeechDetection

            logger.info(
                "Loading pyannote OverlappedSpeechDetection (onset=%.4f)",
                self._osd_onset,
            )
            seg_model = Model.from_pretrained(
                "pyannote/segmentation-3.0",
                use_auth_token=self.hf_token,
            )
            self._osd = OverlappedSpeechDetection(segmentation=seg_model)
            self._osd.instantiate(
                {
                    "min_duration_on": 0.0,
                    "min_duration_off": 0.0,
                }
            )
            self._osd.initialize()
            _dev = self.device if ":" in self.device else "cuda:0"
            if self.device.startswith("cuda"):
                self._osd.to(torch.device(_dev))
        return self._osd

    def transcribe(self, audio_path: str, language: str = None) -> dict:
        """Run faster-whisper and return a whisperx-compatible result dict.

        whisperx.align expects ``{"segments": [...], "language": "..."}`` with
        each segment carrying ``start``/``end``/``text``. We produce exactly
        that shape here so the alignment step downstream is a drop-in.
        """
        lang_arg = language if language else None
        # When auto-detecting, nudge the decoder toward Simplified Chinese.
        # faster-whisper sometimes outputs Traditional Chinese for Mandarin
        # audio; the initial_prompt shifts the prior without forcing it —
        # English/Japanese/etc. audio is unaffected.
        initial_prompt = (
            "以下是普通话的对话，请以简体中文输出。" if lang_arg is None else None
        )
        logger.info(
            "Starting faster-whisper transcription (language=%s)",
            lang_arg or "auto",
        )

        segments_iter, info = self.whisper.transcribe(
            audio_path,
            language=lang_arg,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
            initial_prompt=initial_prompt,
        )
        segments = [
            {
                "start": round(float(s.start), 3),
                "end": round(float(s.end), 3),
                "text": s.text.strip(),
            }
            for s in segments_iter
        ]
        detected = info.language
        logger.info(
            "Transcription done: %d segments, language=%s",
            len(segments),
            detected,
        )
        return {"segments": segments, "language": detected}

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

    def detect_overlaps(self, audio_path: str, onset: float = None) -> dict:
        """Return overlap intervals and statistics.

        If *onset* differs from the current cached onset, the OSD pipeline
        cache is invalidated so it is re-instantiated with the new threshold.
        """
        if onset is not None and onset != self._osd_onset:
            self._osd_onset = onset
            self._osd = None

        osd_result = self.osd_pipeline({"audio": audio_path})
        intervals = [
            (round(seg.start, 3), round(seg.end, 3))
            for seg, _, _ in osd_result.itertracks(yield_label=True)
        ]

        import torchaudio as _torchaudio

        info = _torchaudio.info(audio_path)
        total_s = round(info.num_frames / info.sample_rate, 3)
        overlap_s = round(sum(e - s for s, e in intervals), 3)
        ratio = round(overlap_s / total_s, 4) if total_s > 0 else 0.0
        return {
            "intervals": intervals,
            "total_s": total_s,
            "overlap_s": overlap_s,
            "ratio": ratio,
            "count": len(intervals),
        }

    def separate_overlaps(self, audio_path: str, n_speakers: int = 2) -> list[str]:
        """Run MossFormer2 speech separation. Returns list of separated WAV paths."""
        import torch
        from pathlib import Path

        # Determine the target CUDA device index once.
        _device_idx = (
            (int(self.device.split(":")[-1]) if (":" in self.device) else 0)
            if self.device.startswith("cuda")
            else None
        )

        if getattr(self, "_clearvoice", None) is None:
            from clearvoice import ClearVoice
            from clearvoice.networks import SpeechModel

            # ClearVoice's SpeechModel.__init__ calls get_free_gpu() (nvidia-smi)
            # to select the GPU with the most free memory.  nvidia-smi is a
            # subprocess and ignores CUDA_VISIBLE_DEVICES, so on multi-GPU hosts
            # it returns physical GPU indices.  After seg 0 consumes memory on
            # GPU 0, subsequent calls pick GPU 1 and do set_device(1), but the
            # already-loaded model weights are on GPU 0 → "cuda:0 vs cuda:1".
            #
            # Fix: monkey-patch get_free_gpu on the class to always return the
            # target device index.  This is applied before ClearVoice() is
            # instantiated so every SpeechModel inside it uses the pinned GPU.
            # The patch is permanent for the process lifetime, which is safe
            # because the rest of the pipeline already addresses self.device.
            if _device_idx is not None:
                _pinned = _device_idx
                SpeechModel.get_free_gpu = lambda self: _pinned  # type: ignore[method-assign]
                torch.cuda.set_device(_device_idx)
                logger.info(
                    "separate_overlaps: patched SpeechModel.get_free_gpu → %s "
                    "to prevent multi-GPU tensor scatter in ClearVoice",
                    _device_idx,
                )

            self._clearvoice = ClearVoice(
                task="speech_separation",
                model_names=["MossFormer2_SS_16K"],
            )

        # ClearVoice writes output to a MossFormer2_SS_16K subdirectory inside
        # output_path, naming files "{stem}_s{i}.wav".
        out_dir = str(Path(audio_path).parent)
        stem = Path(audio_path).stem
        cv_subdir = Path(out_dir) / "MossFormer2_SS_16K"

        # Check if separated files already exist from a previous run — skip GPU
        # inference if so to avoid redundant heavy computation.
        preexisting = [cv_subdir / f"{stem}_s{i}.wav" for i in range(1, n_speakers + 1)]
        if all(p.exists() for p in preexisting):
            logger.info("separate_overlaps: reusing cached separation for %s", stem)
            return [str(p) for p in preexisting]

        self._clearvoice(input_path=audio_path, online_write=True, output_path=out_dir)
        results = []
        for i in range(1, n_speakers + 1):
            p = cv_subdir / f"{stem}_s{i}.wav"
            if p.exists():
                results.append(str(p))
        return results

    def separate_overlap_segments(
        self,
        audio_path: str,
        intervals: list[tuple[float, float]],
        min_duration: float = 0.5,
        language: str = None,
    ) -> list[dict]:
        """Run MossFormer2 on each detected overlap interval separately.

        Segment-level separation avoids the dominant-speaker collapse that occurs
        when the full file is fed to the model: within each overlap window both
        speakers are simultaneously active so their energy is balanced.

        Returns list of:
            {start, end, tracks: [{track: 1|2, segments: [...], n_segs: int}]}
        Only intervals longer than *min_duration* seconds are processed.
        """
        import tempfile
        from pathlib import Path
        import torchaudio as _ta

        waveform, sr = _ta.load(audio_path)
        if sr != 16000:
            waveform = _ta.functional.resample(waveform, sr, 16000)
            sr = 16000
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        results = []
        with tempfile.TemporaryDirectory(prefix="vos_sep_") as tmp:
            tmp_path = Path(tmp)
            for idx, (start, end) in enumerate(intervals):
                dur = end - start
                if dur < min_duration:
                    continue

                start_f = int(start * sr)
                end_f = int(end * sr)
                chunk = waveform[:, start_f:end_f]
                if chunk.shape[1] < int(min_duration * sr):
                    continue

                chunk_path = str(tmp_path / f"seg_{idx:04d}.wav")
                _ta.save(chunk_path, chunk, sr)

                try:
                    sep_paths = self.separate_overlaps(chunk_path, n_speakers=2)
                except Exception as exc:
                    logger.warning("separate_overlaps failed for seg %d: %s", idx, exc)
                    continue

                tracks = []
                for i, sp in enumerate(sep_paths, start=1):
                    sp_wave, sp_sr = _ta.load(sp)
                    rms = float(sp_wave.pow(2).mean().sqrt())
                    if rms < 1e-4:
                        tracks.append({"track": i, "segments": [], "n_segs": 0})
                        continue
                    tr = self.transcribe(sp, language=language)
                    tracks.append(
                        {
                            "track": i,
                            "segments": tr["segments"],
                            "n_segs": len(tr["segments"]),
                        }
                    )

                results.append(
                    {"start": round(start, 3), "end": round(end, 3), "tracks": tracks}
                )
                logger.info(
                    "separate_overlap_segments: seg %d [%.2f-%.2f] → %d tracks",
                    idx,
                    start,
                    end,
                    len(tracks),
                )

        return results

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
        language = transcription_result.get("language") or "zh"
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
        raw_audio_path: str = None,
        language: str = None,
        min_speakers: int = None,
        max_speakers: int = None,
        detect_overlap: bool = False,
        separate_speech: bool = False,
        osd_onset: float = 0.5,
    ) -> dict:
        """Full pipeline: transcribe → diarize → forced-align → extract embeddings.

        audio_path      — cleaned/denoised audio fed to Whisper and pyannote.
        raw_audio_path  — original unprocessed audio for voiceprint extraction.
                          Falls back to audio_path when not provided.
        """
        embed_path = raw_audio_path or audio_path

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

        logger.info("Extracting speaker embeddings from %s", embed_path)
        embeddings = self.extract_speaker_embeddings(embed_path, turns)
        logger.info("Extracted embeddings for %d speakers", len(embeddings))

        overlap_result = None
        if detect_overlap:
            logger.info("Running overlap speech detection (onset=%.4f)", osd_onset)
            overlap_result = self.detect_overlaps(audio_path, onset=osd_onset)
            logger.info(
                "OSD found %d overlap intervals (overlap_s=%.3f, ratio=%.4f)",
                overlap_result["count"],
                overlap_result["overlap_s"],
                overlap_result["ratio"],
            )
            for seg in aligned:
                mid = (seg["start"] + seg["end"]) / 2
                seg["has_overlap"] = any(
                    s <= mid <= e for s, e in overlap_result["intervals"]
                )

        overlap_stats = None
        if overlap_result is not None:
            overlap_stats = {
                "total_s": overlap_result["total_s"],
                "overlap_s": overlap_result["overlap_s"],
                "ratio": overlap_result["ratio"],
                "count": overlap_result["count"],
            }

        separated_transcripts: list[dict] = []
        if separate_speech:
            logger.info("Running MossFormer2 speech separation on %s", audio_path)
            separated_paths = self.separate_overlaps(audio_path)
            logger.info("Separation done: %d tracks", len(separated_paths))
            for i, sep_path in enumerate(separated_paths):
                logger.info("Transcribing separated track %d: %s", i + 1, sep_path)
                sep_result = self.transcribe(sep_path, language=language)
                separated_transcripts.append(
                    {
                        "track": i + 1,
                        "path": sep_path,
                        "segments": sep_result["segments"],
                    }
                )

        return {
            "segments": aligned,
            "speaker_embeddings": embeddings,
            "unique_speakers": list(embeddings.keys()),
            "overlap_stats": overlap_stats,
            "separated_tracks": separated_transcripts,
        }
