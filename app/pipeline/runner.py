"""Stable pipeline runner that wires stage slots to current implementations."""

from __future__ import annotations

import logging
from typing import Any

from infra.audio import cleanup_generated_files

from .contracts import PipelineContext, PipelineRequest
from .registry import available_stage_slots, resolve_stage

logger = logging.getLogger(__name__)

DEFAULT_STAGE_ORDER = available_stage_slots()


class PipelineRunner:
    """Execute the stable stage order against the current pipeline implementation."""

    def __init__(
        self,
        stage_order: tuple[str, ...] | None = None,
        stage_overrides: dict[str, Any] | None = None,
    ):
        self.stage_order = tuple(stage_order or DEFAULT_STAGE_ORDER)
        self.stage_overrides = dict(stage_overrides or {})

    def resolve_stage(self, stage_name: str):
        return self.stage_overrides.get(stage_name) or resolve_stage(stage_name)

    def build_context(self, pipeline: Any, request: PipelineRequest) -> PipelineContext:
        return PipelineContext(pipeline=pipeline, request=request)

    def run_context(self, pipeline: Any, request: PipelineRequest) -> PipelineContext:
        context = self.build_context(pipeline, request)
        try:
            for stage_name in self.stage_order:
                logger.info("Running pipeline stage: %s", stage_name)
                stage = self.resolve_stage(stage_name)
                context.mark_stage(stage_name)
                context.metadata.setdefault("selected_providers", {})[
                    stage_name
                ] = request.provider_for(stage_name)
                stage(context)
            return context
        finally:
            cleanup_generated_files(context.temporary_paths)

    def run(self, pipeline: Any, request: PipelineRequest) -> dict[str, Any]:
        return self.run_context(pipeline, request).to_result()


__all__ = [
    "DEFAULT_STAGE_ORDER",
    "PipelineContext",
    "PipelineRequest",
    "PipelineRunner",
]
