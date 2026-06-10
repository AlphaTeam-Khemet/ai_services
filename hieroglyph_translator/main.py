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

import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# ── Load .env before anything else ───────────────────────────────────────────
load_dotenv()

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("khemet.hieroglyph")

# ── Ensure the project root is importable ────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from services.detection_service import load_model
from routers.translation_router import router as translation_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan hook: warms up the YOLO model so the first request is not slow.
    """
    logger.info("🚀 Starting KHEMET Hieroglyph Translator server ...")
    t0 = time.time()
    load_model()
    logger.info("✅ YOLO model warm in %.1fs", time.time() - t0)
    yield
    logger.info("🛑 Shutting down KHEMET Hieroglyph Translator server.")


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="KHEMET Hieroglyph Translator API",
    description=(
        "YOLOv11 hieroglyph detection + LLM translation pipeline "
        "for the KHEMET Egyptian artifact tourism platform."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(translation_router, prefix="/api/v1/hieroglyph")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    """Service health check."""
    return {"status": "ok", "service": "hieroglyph_translator"}


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8002)),
        reload=False,
    )
