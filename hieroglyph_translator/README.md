# KHEMET Hieroglyph Translator Service

The `hieroglyph_translator` is an internal microservice inside the KHEMET Egyptian artifact tourism platform. It provides a FastAPI REST API for detecting Egyptian hieroglyphs in images and translating them into English.

## Architecture

This service acts as an orchestrator for a two-stage machine learning pipeline:

1.  **Stage 1: Detection (YOLOv11)**
    *   Takes an uploaded image (JPEG/PNG/WebP).
    *   Runs a locally hosted YOLOv11 model (`best_V2.pt`).
    *   Detects hieroglyphs and returns their Gardiner codes (e.g., `G17`, `N35`).
    *   *Sorting:* Sorts the detected symbols into left-to-right, top-to-bottom reading order based on bounding box geometry.

2.  **Stage 2: Translation (LLM Bridge)**
    *   Takes the sorted sequence of Gardiner codes.
    *   Makes an HTTP POST request to the internal `chatbot_LLM` service (the `/api/v1/llm/translate-hieroglyphs` endpoint).
    *   The LLM (e.g., LLaMA) returns a concise English translation and a confidence note.

## Endpoints

*   **`POST /api/v1/hieroglyph/translate`**: Runs the full pipeline. Accepts a multipart form with an `image` file and an optional `context_hint` (e.g., "tomb inscription"). Returns a `HieroglyphPipelineResponse` containing the detected symbols, the sequence, and the translation.
*   **`POST /api/v1/hieroglyph/detect-only`**: Runs only Stage 1 (YOLO detection). Useful if you only need the bounding boxes and Gardiner codes without incurring the latency of the LLM call. Returns a `DetectionResult`.
*   **`GET /health`**: Standard health check endpoint.

## Project Structure

```
hieroglyph_translator/
├── main.py                         # FastAPI application entry point
├── requirements.txt                # Python dependencies
├── .env.example                    # Example environment variables
├── Dockerfile                      # Docker image definition
├── .dockerignore                   # Files to exclude from Docker build
├── models/
│   └── best_V2.pt                  # YOLOv11 weights (NOT in Git, mounted at runtime)
├── routers/
│   ├── __init__.py
│   └── translation_router.py       # API routes definition
├── schemas/
│   └── __init__.py                 # Pydantic models for request/response
├── services/
│   ├── __init__.py
│   ├── detection_service.py        # YOLO inference logic
│   ├── llm_bridge_service.py       # HTTP client for chatbot_LLM
│   └── pipeline_service.py         # Orchestrates detection -> translation
├── utils/
│   ├── __init__.py
│   ├── gardiner_lookup.py          # Gardiner code to English label mapping (805 classes)
│   └── ordering.py                 # Reading order sorting algorithm
└── tests/
    ├── __init__.py
    └── test_pipeline.py            # Pytest smoke tests
```

## Running Locally (Docker)

This service is designed to be run as part of the broader KHEMET `docker-compose.yml` stack.

1.  Ensure you have the YOLO weights file located at `AI_services/hieroglyph_translator/model/best_V2.pt`.
2.  Run the stack from the project root:
    ```bash
    docker-compose up --build hieroglyph-translator
    ```

## Constraints and Notes

*   **No External APIs:** All inference (both YOLO and LLM) happens locally within the Docker network.
*   **Client Communication:** Mobile/Web clients **do not** call this service directly. They communicate with the Node.js `backend`, which acts as an API gateway and proxies requests to this service.
*   **Gardiner Classes:** The system supports 805 distinct Gardiner hieroglyph classes, mapped in `utils/gardiner_lookup.py`.
