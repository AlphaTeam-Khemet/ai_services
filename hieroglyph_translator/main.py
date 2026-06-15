#!/usr/bin/env python3
"""
main.py — KHEMET Hieroglyph Translator FastAPI Server
──────────────────────────────────────────────────────
Endpoints:
    POST /api/v1/hieroglyph/translate      → full pipeline (YOLO + LLM translation)
    POST /api/v1/hieroglyph/detect-only   → YOLO detection only
    GET  /health                           → service health check

Run:
    uvicorn main:app --host 0.0.0.0 --port 8002 --reload
"""

import os
import signal
import sys
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# ── Load .env before anything else ───────────────────────────────────────────
load_dotenv()

# ── Ensure the project root is importable ────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from middleware.request_id import RequestIDMiddleware
from utils.logger import get_logger
from utils.startup import validate_env
from services.detection_service import load_model
from routers.translation_router import router as translation_router

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan hook: validates env then warms up the YOLO model
    so the first request is not slow.
    """
    # ── Startup ───────────────────────────────────────────────────────────────
    validate_env()
    logger.info("Starting KHEMET Hieroglyph Translator server")
    t0 = time.time()
    load_model()
    logger.info("YOLO model warm", extra={"elapsed_s": round(time.time() - t0, 1)})
    yield
    # ── Shutdown ──────────────────────────────────────────────────────────────
    logger.info("Shutting down KHEMET Hieroglyph Translator server")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="KHEMET Hieroglyph Translator API",
    description=(
        "YOLOv11 hieroglyph detection + LLM translation pipeline "
        "for the KHEMET Egyptian artifact tourism platform."
    ),
    version="1.0.0",
    lifespan=lifespan,
    # Disable interactive docs in production
    docs_url="/docs" if os.getenv("ENV") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENV") != "production" else None,
)

# ── Middleware ────────────────────────────────────────────────────────────────

# Request ID — must be registered first
app.add_middleware(RequestIDMiddleware)

# CORS — read from env, never hardcode *
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

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(translation_router, prefix="/api/v1/hieroglyph")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Service health check."""
    return {
        "status": "ok",
        "service": "hieroglyph_translator",
        "port": 8002,
        "version": "1.0.0",
    }


# ── Graceful Shutdown ─────────────────────────────────────────────────────────
def _handle_shutdown(signum, frame):
    logger.info("Shutdown signal received", extra={"signal": signum})
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8002)),
        reload=False,
        access_log=False,  # Logging is handled via structured middleware
    )
