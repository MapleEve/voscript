"""Default provider for assembling and persisting pipeline artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from config import DENOISE_MODEL, DENOISE_SNR_THRESHOLD
from infra.transcription_artifacts import persist_transcription_artifacts
from pipeline.contracts import PipelineContext, PipelineResult


class InMemoryArtifactsProvider:
    """Assemble the final transcript payload from the current context state."""

    @staticmethod
    def _build_cluster_remap(
        speaker_map: dict[str, dict[str, object]],
    ) -> dict[str, str]:
        id_to_clusters: dict[str, list[tuple[str, float]]] = {}
        for speaker_label, info in speaker_map.items():
            matched_id = info.get("matched_id")
            similarity = float(info.get("similarity", 0) or 0)
            if matched_id is not None:
                id_to_clusters.setdefault(str(matched_id), []).append(
                    (speaker_label, similarity)
                )

        cluster_remap: dict[str, str] = {}
        for cluster_list in id_to_clusters.values():
            cluster_list.sort(key=lambda item: item[1], reverse=True)
            canonical_label = cluster_list[0][0]
            for speaker_label, _similarity in cluster_list[1:]:
                cluster_remap[speaker_label] = canonical_label
        return cluster_remap

    @staticmethod
    def _build_segments(
        aligned_segments: list[dict],
        speaker_map: dict[str, dict],
    ) -> tuple[list[dict], list[str]]:
        cluster_remap = InMemoryArtifactsProvider._build_cluster_remap(speaker_map)
        segments: list[dict] = []
        seen_speakers: set[str] = set()
        unique_speakers: list[str] = []

        for index, segment in enumerate(aligned_segments):
            speaker_label = segment["speaker"]
            canonical_label = cluster_remap.get(speaker_label, speaker_label)
            match = speaker_map.get(canonical_label, speaker_map.get(speaker_label, {}))
            speaker_name = match.get("matched_name", canonical_label)
            output = {
                "id": index,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"],
                "speaker_label": canonical_label,
                "speaker_id": match.get("matched_id"),
                "speaker_name": speaker_name,
                "similarity": match.get("similarity", 0),
            }
            if segment.get("words"):
                output["words"] = segment["words"]
            segments.append(output)

            if speaker_name not in seen_speakers:
                seen_speakers.add(speaker_name)
                unique_speakers.append(speaker_name)

        return segments, unique_speakers

    def _build_transcription(self, context: PipelineContext) -> dict | None:
        if context.request.artifact_dir is None:
            return None

        effective_denoise = (
            (context.request.denoise_model or DENOISE_MODEL).strip().lower()
        )
        effective_snr = (
            context.request.snr_threshold
            if context.request.snr_threshold is not None
            else DENOISE_SNR_THRESHOLD
        )
        segments, unique_speakers = self._build_segments(
            context.aligned_segments,
            context.voiceprint_matches,
        )
        warning = None
        if not context.voiceprint_matches and not context.speaker_embeddings:
            warning = "no_speakers_detected"

        transcription = {
            "id": context.request.artifact_dir.name,
            "filename": Path(context.request.audio_path).name,
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
            "status": "completed",
            "language": context.request.language,
            "segments": segments,
            "speaker_map": context.voiceprint_matches,
            "unique_speakers": unique_speakers,
            "params": {
                "language": context.request.language or "auto",
                "denoise_model": effective_denoise,
                "snr_threshold": effective_snr,
                "voiceprint_threshold": context.request.voiceprint_threshold,
                "min_speakers": context.request.min_speakers,
                "max_speakers": context.request.max_speakers,
                "no_repeat_ngram_size": context.request.no_repeat_ngram_size or 0,
            },
        }
        if warning is not None:
            transcription["warning"] = warning
        return transcription

    def build(self, context: PipelineContext) -> PipelineResult:
        transcription = self._build_transcription(context)
        artifact_paths = None
        if transcription is not None and context.request.artifact_dir is not None:
            persisted = persist_transcription_artifacts(
                context.request.artifact_dir,
                transcription,
                context.speaker_embeddings,
            )
            artifact_paths = {
                "result_path": str(persisted.result_path),
                "embedding_paths": {
                    label: str(path)
                    for label, path in persisted.embedding_paths.items()
                },
            }
            segments = transcription["segments"]
            unique_speakers = transcription["unique_speakers"]
        else:
            segments = context.aligned_segments
            unique_speakers = list(context.speaker_embeddings.keys())

        return PipelineResult(
            segments=segments,
            speaker_embeddings=context.speaker_embeddings,
            unique_speakers=unique_speakers,
            transcription=transcription,
            artifact_paths=artifact_paths,
        )


default_artifacts_provider = InMemoryArtifactsProvider()


__all__ = [
    "InMemoryArtifactsProvider",
    "default_artifacts_provider",
]
