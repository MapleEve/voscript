"""Default provider for assembling and persisting pipeline artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from config import DENOISE_MODEL, DENOISE_SNR_THRESHOLD
from infra.audio.paths import safe_speaker_label
from infra.transcription_artifacts import persist_transcription_artifacts
from postprocess.segments import build_display_names, build_result_segments
from providers.kernel_bridge import (
    artifact_manifest_contract,
    postprocess_segments,
    rust_provider_paths_enabled,
)
from pipeline.contracts import (
    ArtifactManifestEntry,
    PipelineContext,
    PipelineResult,
    build_artifact_manifest,
)


class InMemoryArtifactsProvider:
    """Assemble the final transcript payload from the current context state."""

    @staticmethod
    def _build_display_names(
        speaker_labels: list[str],
        speaker_map: dict[str, dict],
    ) -> dict[str, str]:
        return build_display_names(speaker_labels, speaker_map)

    @staticmethod
    def _build_segments(
        aligned_segments: list[dict],
        speaker_map: dict[str, dict],
    ) -> tuple[list[dict], list[str]]:
        if rust_provider_paths_enabled():
            response = postprocess_segments(
                {
                    "aligned_segments": aligned_segments,
                    "speaker_map": speaker_map,
                }
            )
            return response["segments"], response["unique_speakers"]
        return build_result_segments(aligned_segments, speaker_map)

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
        embedding_labels = sorted(context.speaker_embeddings)
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
            "artifacts": self._build_artifact_manifest(embedding_labels),
        }
        if context.transcription_result is not None:
            guard_report = context.transcription_result.get("hallucination_guard")
            if guard_report is not None:
                transcription["asr_hallucination_guard"] = guard_report
        alignment_metadata = context.metadata.get("diarization", {}).get("alignment")
        if alignment_metadata:
            transcription["alignment"] = alignment_metadata
        if warning is not None:
            transcription["warning"] = warning
        return transcription

    @staticmethod
    def _build_artifact_manifest(speaker_labels: list[str]) -> dict:
        stable = [
            ArtifactManifestEntry(
                name="result",
                filename="result.json",
                role="primary_result",
                media_type="application/json",
                required_for_result=True,
            )
        ]
        stable.extend(
            ArtifactManifestEntry(
                name="speaker_embedding",
                filename=f"emb_{safe_speaker_label(speaker_label)}.npy",
                role="speaker_embedding",
                media_type="application/octet-stream",
                speaker_label=speaker_label,
            )
            for speaker_label in speaker_labels
        )
        manifest = build_artifact_manifest(stable=stable)
        if rust_provider_paths_enabled():
            return artifact_manifest_contract(manifest)
        return manifest

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
