"""
Egyptian Artifact Recognition — FastAPI Application
=====================================================
A lightweight REST API backed by CLIP zero-shot image classification.

Endpoints:
    GET  /health          Service liveness check
    GET  /debug/model-info  CLIP model metadata
    POST /predict         Classify an uploaded image
    POST /translate       Alias for /predict (hieroglyph translation flow)
"""

import os
import signal
import sys
from contextlib import asynccontextmanager

# ── Ensure project root (CV_Recognition/) is on sys.path ─────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from middleware.request_id import RequestIDMiddleware
from utils.logger import get_logger
from utils.startup import validate_env
from model.predict import get_model_info, predict_image

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    validate_env()
    logger.info("CV Recognition service started (CLIP Zero-Shot only)", extra={"port": 8000})
    yield
    logger.info("CV Recognition service shutting down")


app = FastAPI(
    title="Egyptian Artifact Recognition API",
    description="Upload an image and get the predicted artifact class via CLIP zero-shot classification.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENV") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENV") != "production" else None,
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(RequestIDMiddleware)

allowed_origins = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)


# ── Graceful Shutdown ─────────────────────────────────────────────────────────
def _handle_shutdown(signum, frame):
    logger.info("Shutdown signal received", extra={"signal": signum})
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    # NOTE: Do NOT call is_model_ready() here — CLIP model loading is slow
    # (~30-120s on first pull) and would cause health checks to fail during
    # startup. The service is "healthy" as soon as uvicorn is accepting requests.
    # Use GET /debug/model-info to check whether the model is loaded.
    return {
        "status": "ok",
        "service": "cv_recognition",
        "port": 8000,
        "version": "2.0.0",
        "model_type": "CLIP Zero-Shot",
    }


@app.get("/debug/model-info")
async def debug_model_info():
    """Return CLIP model metadata and candidate label list."""
    return get_model_info()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept an uploaded image and return the predicted class name with confidence.

    **Request:** `multipart/form-data` with a field named `file`.

    **Response:**
    ```json
    {
      "class_name": "Mask_of_Tutankhamun",
      "confidence": 0.87,
      "top_predictions": [...]
    }
    ```
    """
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image. Received: {file.content_type}",
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = predict_image(image_bytes)
        logger.info(
            "Prediction complete",
            extra={"class_name": result.get("class_name"), "confidence": result.get("confidence")},
        )
        return result

    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("Prediction service unavailable", extra={"error": str(e)})
        raise HTTPException(status_code=503, detail=f"CLIP model is unavailable: {str(e)}")
    except Exception as e:
        logger.error("Prediction failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/translate")
async def translate(file: UploadFile = File(...)):
    """Alias for /predict — used by the hieroglyph translation flow."""
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail=f"File must be an image. Received: {file.content_type}")

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        result = predict_image(image_bytes)
        result["note"] = (
            "CLIP zero-shot classification result. "
            "A dedicated hieroglyph translation model is not yet integrated."
        )
        return result
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("Translation service unavailable", extra={"error": str(e)})
        raise HTTPException(status_code=503, detail=f"CLIP model is unavailable: {str(e)}")
    except Exception as e:
        logger.error("Translation failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
