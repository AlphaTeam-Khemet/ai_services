"""
services/pipeline_service.py — orchestrates detection → translation pipeline.
"""

from __future__ import annotations

import logging
import uuid

from schemas import HieroglyphPipelineResponse, DetectionResult
from services.detection_service import detect_symbols
from services.llm_bridge_service import request_translation

logger = logging.getLogger("khemet.hieroglyph.pipeline")


async def run_pipeline(
    image_bytes: bytes,
    context_hint: str | None = None,
) -> HieroglyphPipelineResponse:
    """Run the full hieroglyph detection + translation pipeline.

    Steps:
    1. YOLO detection → :class:`~schemas.DetectionResult`
    2. If symbols found, call LLM bridge → :class:`~schemas.TranslationResponse`
    3. Return :class:`~schemas.HieroglyphPipelineResponse`

    Args:
        image_bytes: Raw image bytes.
        context_hint: Optional free-text context passed to the LLM.

    Returns:
        Full pipeline response including image_id, detection, and translation.
    """
    image_id = uuid.uuid4().hex[:8]
    logger.info("Pipeline start [image_id=%s]", image_id)

    # ── Stage 1: Detection ────────────────────────────────────────────────────
    detection: DetectionResult = await detect_symbols(image_bytes)

    # ── Stage 2: Early exit if no symbols detected ────────────────────────────
    if not detection.symbol_sequence:
        logger.info(
            "Pipeline [image_id=%s]: no symbols detected — skipping LLM call.",
            image_id,
        )
        return HieroglyphPipelineResponse(
            image_id=image_id,
            detection=detection,
            translation=None,
        )

    # ── Stage 3: Translation ──────────────────────────────────────────────────
    logger.info(
        "Pipeline [image_id=%s]: calling LLM for %d symbols.",
        image_id,
        len(detection.symbol_sequence),
    )
    translation = await request_translation(
        symbol_sequence=detection.symbol_sequence,
        context_hint=context_hint,
    )

    logger.info("Pipeline [image_id=%s] complete.", image_id)

    return HieroglyphPipelineResponse(
        image_id=image_id,
        detection=detection,
        translation=translation,
    )
