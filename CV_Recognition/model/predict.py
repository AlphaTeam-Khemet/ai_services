"""
Prediction utilities for the Egyptian artifact recognition API.

Uses CLIP zero-shot image classification (openai/clip-vit-large-patch14) as the
sole inference engine. No model.h5 or TensorFlow is required.
"""

import io
import json
import logging
import os

import torch
from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())

_MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
_CLIP_CANDIDATE_LABELS_PATH = os.path.join(_MODEL_DIR, "clip_candidate_labels.json")
_CLASS_NAMES_PATH = os.path.join(_MODEL_DIR, "class_names.json")

# ── Device selection ──────────────────────────────────────────────────────────
_DEVICE = 0 if torch.cuda.is_available() else -1  # 0 = first GPU, -1 = CPU
_DEVICE_NAME = f"cuda:{_DEVICE}" if _DEVICE >= 0 else "cpu"

# ── Candidate labels ──────────────────────────────────────────────────────────
with open(_CLASS_NAMES_PATH, "r", encoding="utf-8") as _f:
    _class_names: list[str] = json.load(_f)

if os.path.isfile(_CLIP_CANDIDATE_LABELS_PATH):
    with open(_CLIP_CANDIDATE_LABELS_PATH, "r", encoding="utf-8") as _f:
        _clip_candidate_labels: list[str] = json.load(_f)
else:
    _clip_candidate_labels = [name.replace("_", " ") for name in _class_names]

# ── Lazy-loaded CLIP pipeline ─────────────────────────────────────────────────
_clip_pipeline = None
_clip_load_error: str | None = None

_CLIP_MODEL = os.getenv("CLIP_MODEL", "openai/clip-vit-large-patch14")


def _ensure_clip_loaded() -> None:
    global _clip_pipeline, _clip_load_error
    if _clip_pipeline is not None:
        return
    try:
        logger.info(
            "Loading CLIP Zero-Shot model (%s) on %s...", _CLIP_MODEL, _DEVICE_NAME.upper()
        )
        _clip_pipeline = pipeline(
            "zero-shot-image-classification",
            model=_CLIP_MODEL,
            device=_DEVICE,  # 0 = GPU, -1 = CPU
        )
        _clip_load_error = None
        logger.info("CLIP Zero-Shot model loaded successfully on %s.", _DEVICE_NAME.upper())
    except Exception as exc:
        _clip_load_error = str(exc)
        logger.error("Failed to load CLIP Zero-Shot model: %s", _clip_load_error)
        raise RuntimeError(f"CLIP model failed to load: {_clip_load_error}") from exc


def is_model_ready() -> bool:
    """Return True if the CLIP pipeline is loaded and ready."""
    try:
        _ensure_clip_loaded()
        return _clip_pipeline is not None
    except Exception:
        return False


def get_model_info() -> dict:
    """Return metadata about the active CLIP model."""
    ready = is_model_ready()
    return {
        "model_type": "CLIP Zero-Shot",
        "clip_model": _CLIP_MODEL,
        "model_ready": ready,
        "model_load_error": _clip_load_error,
        "number_of_candidates": len(_clip_candidate_labels),
        "candidate_labels": _clip_candidate_labels,
    }


def predict_image(image_bytes: bytes) -> dict:
    """
    Run CLIP zero-shot classification on the supplied image bytes.

    Returns a dict with:
        class_name  – top predicted label (underscored)
        confidence  – probability of the top label (0-1)
        top_predictions – list of top-3 {class_name, confidence} dicts
    """
    _ensure_clip_loaded()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    raw_results = _clip_pipeline(image, candidate_labels=_clip_candidate_labels)

    top_predictions = [
        {
            # Keep the label exactly as CLIP returns it (human-readable with spaces).
            # The backend classMapping.js matches against these exact strings.
            "class_name": res["label"],
            "confidence": round(res["score"], 4),
        }
        for res in raw_results[:3]
    ]

    logger.info("CLIP top-3 predictions: %s", top_predictions)

    return {
        "class_name": top_predictions[0]["class_name"],
        "confidence": top_predictions[0]["confidence"],
        "top_predictions": top_predictions,
    }
