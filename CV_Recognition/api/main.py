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

import sys
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("khemet.cv")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from model.predict import get_model_info, predict_image, is_model_ready

app = FastAPI(
    title="Egyptian Artifact Recognition API",
    description="Upload an image and get the predicted artifact class.",
    version="1.0.0",
)

# Read CORS origins from environment — defaults to * (allow all).
# In production set: ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model_ready": is_model_ready()}


@app.get("/debug/model-info")
async def debug_model_info():
    return get_model_info()


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
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
        return result

    except HTTPException:
        raise
    except Exception as e:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")
