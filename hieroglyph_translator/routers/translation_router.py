"""
routers/translation_router.py — FastAPI router for hieroglyph endpoints.

Mounted at /api/v1/hieroglyph in main.py.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from schemas import DetectionResult, HieroglyphPipelineResponse
from services.detection_service import detect_symbols
from services.pipeline_service import run_pipeline

logger = logging.getLogger("khemet.hieroglyph.router")

router = APIRouter(tags=["hieroglyph"])


@router.post("/translate", response_model=HieroglyphPipelineResponse)
async def translate(
    image: UploadFile = File(..., description="Hieroglyph image (JPEG/PNG/WebP)"),
    context_hint: str | None = Form(None, description="Optional context, e.g. 'tomb inscription'"),
    min_confidence: float = Form(0.33, description="Minimum confidence threshold"),
    reading_direction: str = Form("ltr", description="Reading direction ('ltr' or 'rtl')"),
) -> HieroglyphPipelineResponse:
    """Full pipeline: YOLO detection → LLM translation.

    **Request:** `multipart/form-data`
    - `image`: image file
    - `context_hint` *(optional)*: free-text context

    **Response:** :class:`~schemas.HieroglyphPipelineResponse`
    """
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Received: {image.content_type}",
        )

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = await run_pipeline(image_bytes, context_hint=context_hint, min_confidence=min_confidence, reading_direction=reading_direction)
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Pipeline failed: {exc}")


@router.post("/detect-only", response_model=DetectionResult)
async def detect_only(
    image: UploadFile = File(..., description="Hieroglyph image (JPEG/PNG/WebP)"),
) -> DetectionResult:
    """YOLO detection only — does NOT call the LLM.

    **Request:** `multipart/form-data` with `image` field.

    **Response:** :class:`~schemas.DetectionResult`
    """
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Received: {image.content_type}",
        )

    try:
        image_bytes = await image.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = await detect_symbols(image_bytes)
        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Detection failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Detection failed: {exc}")
