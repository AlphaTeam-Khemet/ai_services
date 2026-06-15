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


def sort_glyphs_by_position(symbols: list[DetectedSymbol], reading_direction: str = "ltr") -> list[DetectedSymbol]:
    """Sort glyphs spatially based on bounding boxes. Top to bottom, then Left/Right."""
    if reading_direction == "rtl":
        return sorted(symbols, key=lambda s: (s.bbox[1], -s.bbox[0]))
    else:
        return sorted(symbols, key=lambda s: (s.bbox[1], s.bbox[0]))

def _run_inference(image_bytes: bytes, min_confidence: float = 0.33, reading_direction: str = "ltr") -> DetectionResult:
    """Synchronous YOLO inference. Must be wrapped with asyncio.to_thread()."""
    global _model
    if _model is None:
        load_model()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_w, img_h = image.size

    # Always use a low threshold for the raw YOLO call so we can deduplicate properly,
    # then filter strictly using min_confidence
    results = _model(image, conf=0.10, verbose=False)

    raw_symbols: list[DetectedSymbol] = []
    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id = int(box.cls[0])
            gardiner_code: str = result.names[cls_id]
            confidence = float(box.conf[0])

            if confidence < min_confidence:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            bbox = [x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h]

            raw_symbols.append(
                DetectedSymbol(
                    gardiner_code=gardiner_code,
                    label=get_symbol_label(gardiner_code),
                    confidence=round(confidence, 4),
                    bbox=[round(v, 6) for v in bbox],
                )
            )

    # Sort spatially first
    sorted_symbols = sort_glyphs_by_position(raw_symbols, reading_direction)

    # Deduplicate: Keep highest confidence detection per unique code, preserving order
    seen = {}
    symbols: list[DetectedSymbol] = []
    
    # Pass 1: Find max confidence for each code
    max_confs = {}
    for s in sorted_symbols:
        if s.gardiner_code not in max_confs or s.confidence > max_confs[s.gardiner_code]:
            max_confs[s.gardiner_code] = s.confidence

    # Pass 2: Keep only the symbol if it is the max confidence one (and only keep it once)
    for s in sorted_symbols:
        if s.gardiner_code not in seen and s.confidence >= max_confs[s.gardiner_code]:
            seen[s.gardiner_code] = True
            symbols.append(s)

    symbol_sequence = [s.gardiner_code for s in symbols]

    logger.info(
        "Detection complete: %d symbols detected (conf≥%.2f) - Reading: %s",
        len(symbols),
        min_confidence,
        reading_direction
    )

    return DetectionResult(symbols=symbols, symbol_sequence=symbol_sequence)


async def detect_symbols(image_bytes: bytes, min_confidence: float = 0.33, reading_direction: str = "ltr") -> DetectionResult:
    """Run YOLO detection asynchronously."""
    return await asyncio.to_thread(_run_inference, image_bytes, min_confidence, reading_direction)
