"""Pipeline metadata ownership and public-surface contract."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class PipelineMetadataEntry:
    """Ownership record for a stable PipelineContext.metadata key or path."""

    owner: str
    writers: tuple[str, ...]
    readers: tuple[str, ...] = ()
    public: bool = False
    allow_overwrite: bool = False
    description: str = ""


PIPELINE_METADATA_CONTROL_KEYS = (
    "executed_stages",
    "selected_providers",
    "provider_capabilities",
    "stage_timings",
)

PIPELINE_METADATA_STAGE_KEYS = (
    "ingest",
    "normalize",
    "enhance",
    "vad",
    "asr",
    "diarization",
    "embedding",
    "voiceprint_match",
    "punc",
    "postprocess",
    "artifacts",
)

PIPELINE_METADATA_TOP_LEVEL_KEYS = (
    *PIPELINE_METADATA_CONTROL_KEYS,
    *PIPELINE_METADATA_STAGE_KEYS,
)

PIPELINE_METADATA_PUBLIC_PATHS = ("diarization.alignment",)

PIPELINE_METADATA_STAGE_WRITERS = {
    "ingest": ("providers.ingest.default",),
    "normalize": ("pipeline.stages.normalize",),
    "enhance": ("pipeline.stages.enhance",),
    "vad": ("providers.vad.default",),
    "asr": ("pipeline.stages.asr",),
    "diarization": ("pipeline.stages.diarization",),
    "embedding": ("pipeline.stages.embedding",),
    "voiceprint_match": ("pipeline.stages.voiceprint_match",),
    "punc": ("providers.punc.default",),
    "postprocess": ("providers.postprocess.default",),
    "artifacts": ("pipeline.stages.artifacts",),
}

PIPELINE_METADATA_CONTRACT: dict[str, PipelineMetadataEntry] = {
    "executed_stages": PipelineMetadataEntry(
        owner="pipeline.runner",
        writers=("PipelineContext.mark_stage",),
        readers=("pipeline.runner",),
        description="Ordered stage names observed by the runner.",
    ),
    "selected_providers": PipelineMetadataEntry(
        owner="pipeline.runner",
        writers=("pipeline.runner",),
        readers=("pipeline.runner",),
        description="Provider selected for each stage before execution.",
    ),
    "provider_capabilities": PipelineMetadataEntry(
        owner="pipeline.runner",
        writers=("pipeline.runner",),
        readers=("pipeline.runner",),
        description="Provider capability preflight metadata keyed by stage.",
    ),
    "stage_timings": PipelineMetadataEntry(
        owner="pipeline.runner",
        writers=("pipeline.runner",),
        readers=("pipeline.runner",),
        description="Elapsed stage timing in seconds keyed by stage.",
    ),
}

PIPELINE_METADATA_CONTRACT.update(
    {
        stage: PipelineMetadataEntry(
            owner=stage,
            writers=("pipeline.runner", *PIPELINE_METADATA_STAGE_WRITERS[stage]),
            readers=("pipeline.runner", "providers.artifacts.default"),
            allow_overwrite=True,
            description=f"Private execution metadata owned by the {stage} stage.",
        )
        for stage in PIPELINE_METADATA_STAGE_KEYS
    }
)

PIPELINE_METADATA_PATH_CONTRACT: dict[str, PipelineMetadataEntry] = {
    "diarization.alignment": PipelineMetadataEntry(
        owner="diarization",
        writers=("pipeline.stages.diarization",),
        readers=("providers.artifacts.default",),
        public=True,
        allow_overwrite=False,
        description="Safe forced-alignment summary allowed in result artifacts.",
    ),
}

PUBLIC_ALIGNMENT_METADATA_KEYS = (
    "status",
    "reason",
    "model",
    "duration_s",
    "max_duration_s",
    "cache_only",
    "device",
)

PublicMetadataValue = str | int | float | bool | None


def _is_public_metadata_scalar(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool, int)):
        return True
    return isinstance(value, float) and math.isfinite(value)


def normalize_public_alignment_metadata(
    value: Any,
) -> dict[str, PublicMetadataValue]:
    """Return only the stable, JSON-safe alignment fields exposed publicly."""

    if not isinstance(value, dict):
        return {}

    normalized: dict[str, PublicMetadataValue] = {}
    for key in PUBLIC_ALIGNMENT_METADATA_KEYS:
        field_value = value.get(key)
        if key in value and _is_public_metadata_scalar(field_value):
            normalized[key] = field_value
    return normalized


__all__ = [
    "PIPELINE_METADATA_CONTRACT",
    "PIPELINE_METADATA_CONTROL_KEYS",
    "PIPELINE_METADATA_PATH_CONTRACT",
    "PIPELINE_METADATA_PUBLIC_PATHS",
    "PIPELINE_METADATA_STAGE_WRITERS",
    "PIPELINE_METADATA_STAGE_KEYS",
    "PIPELINE_METADATA_TOP_LEVEL_KEYS",
    "PUBLIC_ALIGNMENT_METADATA_KEYS",
    "PipelineMetadataEntry",
    "normalize_public_alignment_metadata",
]
