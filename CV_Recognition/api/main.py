"""
Egyptian Artifact Recognition — FastAPI Application
=====================================================
A lightweight REST API with a single endpoint:

    POST /predict
        Accepts an uploaded image file and returns the
        predicted artifact class name with confidence.

CORS is enabled so that a separate frontend can call this API.
No database code is included — only classification.
"""

import os
import signal
import sys
from contextlib import asynccontextmanager

# ── Ensure project root (CV_Recognition/) is on sys.path ─────────────────────
# api/main.py lives one level below the project root. Adding the parent here
# makes utils/ and middleware/ at the CV_Recognition/ level importable.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from middleware.request_id import RequestIDMiddleware
from utils.logger import get_logger
from utils.startup import validate_env
from model.predict import get_model_info, predict_image, is_model_ready

logger = get_logger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    validate_env()
    logger.info("CV Recognition service started", extra={"port": 8000})
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("CV Recognition service shutting down")


app = FastAPI(
    title="Egyptian Artifact Recognition API",
    description="Upload an image and get the predicted artifact class.",
    version="1.0.0",
    lifespan=lifespan,
    # Disable interactive docs in production
    docs_url="/docs" if os.getenv("ENV") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENV") != "production" else None,
)

# ── Middleware ─────────────────────────────────────────────────────────────────

# Request ID — must be registered first
app.add_middleware(RequestIDMiddleware)

# CORS — read from env, never hardcode *
# In production set: ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
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
    return {
        "status": "ok",
        "service": "cv_recognition",
        "port": 8000,
        "version": "1.0.0",
        "model_ready": is_model_ready(),
    }


@app.get("/debug/model-info")
async def debug_model_info():
    return get_model_info()


@app.post("/predict")
async def predict(file: UploadFile = File(...), request: None = None):
    """
    Accept an uploaded image and return the predicted class name
    with a confidence score.

    **Request:** `multipart/form-data` with a field named `file`.

    **Response:**
    ```json
    {
      "class_name": "Mask of Tutankhamun",
      "confidence": 0.94
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
        raise HTTPException(
            status_code=503,
            detail=f"Prediction model is unavailable: {str(e)}",
        )
    except Exception as e:
        logger.error("Prediction failed", extra={"error": str(e)})
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


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
            "Classification-based result. "
            "A dedicated hieroglyph translation model is not yet integrated."
        )
        return result
    except HTTPException:
        raise
    except RuntimeError as e:
        logger.error("Translation service unavailable", extra={"error": str(e)})
        raise HTTPException(status_code=503, detail=f"Prediction model is unavailable: {str(e)}")
    except Exception as e:
        logger.error("Translation failed", extra={"error": str(e)})
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
