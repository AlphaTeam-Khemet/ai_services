"""
services/detection_service.py — YOLOv11 hieroglyph detection.

Singleton model load + async-safe inference via asyncio.to_thread().
"""

from __future__ import annotations

import asyncio
import io
import logging
import os

from PIL import Image

from schemas import DetectedSymbol, DetectionResult
from utils.gardiner_lookup import get_symbol_label
from utils.ordering import sort_symbols_reading_order

logger = logging.getLogger("khemet.hieroglyph.detection")

# ── Singleton state ───────────────────────────────────────────────────────────
_model = None  # ultralytics YOLO instance


def load_model() -> None:
    """Load the YOLO model into the module-level singleton.

    Called once during FastAPI lifespan startup so the first request is fast.
    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _model
    if _model is not None:
        return

    from ultralytics import YOLO  # lazy import — only needed in this service

    model_path = os.getenv("YOLO_MODEL_PATH", "model/best_V2.pt")
    logger.info("Loading YOLO model from '%s' ...", model_path)
    _model = YOLO(model_path)
    logger.info("✅ YOLO model loaded — %d classes", len(_model.names))


def _run_inference(image_bytes: bytes) -> DetectionResult:
    """Synchronous YOLO inference.  Must be wrapped with asyncio.to_thread()."""
    global _model
    if _model is None:
        load_model()

    conf_threshold = float(os.getenv("YOLO_CONF_THRESHOLD", "0.35"))

    # Decode bytes → PIL image
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = image.size

    results = _model(image, conf=conf_threshold, verbose=False)

    symbols: list[DetectedSymbol] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            gardiner_code: str = result.names[cls_id]
            confidence = float(box.conf[0])

            # Absolute pixel coords → normalised [0, 1]
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [
                x1 / img_w,
                y1 / img_h,
                x2 / img_w,
                y2 / img_h,
            ]

            symbols.append(
                DetectedSymbol(
                    gardiner_code=gardiner_code,
                    label=get_symbol_label(gardiner_code),
                    confidence=round(confidence, 4),
                    bbox=[round(v, 6) for v in bbox],
                )
            )

    # Sort into reading order (left→right, top→bottom)
    symbols = sort_symbols_reading_order(symbols)
    symbol_sequence = [s.gardiner_code for s in symbols]

    logger.info(
        "Detection complete: %d symbols detected (conf≥%.2f)",
        len(symbols),
        conf_threshold,
    )

    return DetectionResult(symbols=symbols, symbol_sequence=symbol_sequence)


async def detect_symbols(image_bytes: bytes) -> DetectionResult:
    """Run YOLO detection asynchronously (offloads blocking call to a thread).

    Args:
        image_bytes: Raw image bytes (JPEG / PNG / WebP).

    Returns:
        :class:`~schemas.DetectionResult` with detected symbols in reading order.
    """
    return await asyncio.to_thread(_run_inference, image_bytes)
