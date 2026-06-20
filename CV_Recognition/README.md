# Egyptian Artifact Recognition API

FastAPI service for artifact classification using a **HuggingFace CLIP Zero-Shot** classification model.

- **Framework:** HuggingFace Transformers (PyTorch)
- **Model:** `openai/clip-vit-base-patch32` (configurable via `CLIP_MODEL` env var)
- **Classes:** Dynamically loaded from `model/clip_candidate_labels.json`

## Architecture

1. **Upload:** The client sends an image file via `POST /predict`.
2. **Preprocessing:** The image is decoded and converted to RGB.
3. **CLIP Inference:** The image is classified zero-shot against all candidate labels in `clip_candidate_labels.json` using the CLIP model.
4. **Response:** Returns the top predicted class, confidence score, and a ranked list of top predictions.

> **Note:** The previous dual-model architecture (TensorFlow primary + CLIP fallback) has been consolidated into a single CLIP-only pipeline. There is no `model.h5` file required.

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**API docs:** `http://localhost:8000/docs`

**Health check:**
```http
GET http://localhost:8000/health
```

**Model debug info:**
```http
GET http://localhost:8000/debug/model-info
```

**Prediction:**
```http
POST http://localhost:8000/predict
Content-Type: multipart/form-data
```

**Form field:**
```text
file: <image file>
```

**Example response (Primary Model):**
```json
{
  "class_name": "Golden Mask of Tutankhamun",
  "confidence": 0.9832,
  "top_predictions": [
    {"class_name": "Golden Mask of Tutankhamun", "confidence": 0.9832},
    {"class_name": "Golden Throne of Tutankhamun", "confidence": 0.0104},
    {"class_name": "Egyptian Museum, Cairo", "confidence": 0.0031}
  ],
  "fallback_triggered": false
}
```

**Example response (CLIP Fallback):**
```json
{
  "class_name": "Sphinx",
  "confidence": 0.9912,
  "top_predictions": [
    {"class_name": "Unknown Artifact", "confidence": 0.4320}
  ],
  "fallback_triggered": true,
  "clip_predictions": [
    {"class_name": "Sphinx", "confidence": 0.9912}
  ],
  "message": "Primary model confidence low. CLIP Zero-Shot fallback triggered and used as final result."
}
```

## Run With Docker

From the project root:
```bash
docker compose up -d cv-recognition
```

> **Note:** `model/model.h5` must be present before building the image or mounted as a volume.

## Production Hardening & Middleware

This service has been hardened for production environments. It includes:
- **Request Tracing**: Injects a unique UUID per request using a custom middleware (`middleware/request_id.py`).
- **Structured JSON Logging**: Implements standardized machine-readable logging for precise observability.
- **Environment Validation**: Strict startup assertions to ensure models and environment variables are present.
- **Graceful Shutdown**: Safely closes resources and flushes logs on receiving termination signals.
