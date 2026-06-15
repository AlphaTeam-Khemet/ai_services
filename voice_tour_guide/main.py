"""
main.py
=======
KHEMET Voice Tour Guide Service
================================
Lightweight FastAPI microservice with a single responsibility:
convert artifact description text to MP3 audio using ElevenLabs TTS.

All caching and database operations are handled by the Node.js backend.
This service only generates audio and returns the file URL.

Service Details:
    Port    : 8003
    Primary TTS  : ElevenLabs API (eleven_multilingual_v2)

Endpoints:
    POST /generate              Convert text to MP3 audio
    GET  /health                Service health check
    GET  /static/audio/*.mp3    Serve generated audio files
"""

import os
import signal
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from middleware.request_id import RequestIDMiddleware
from utils.logger import get_logger
from utils.startup import validate_env
from generate import router as generate_router

logger = get_logger(__name__)


# ── Lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────
    validate_env()
    Path("static/audio").mkdir(parents=True, exist_ok=True)
    logger.info("Voice Tour Guide service started", extra={"port": 8003})
    yield
    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("Voice Tour Guide service shutting down")


# ── Application Setup ──────────────────────────────────────────────────────
app = FastAPI(
    title="KHEMET Voice Tour Guide Service",
    description=(
        "Lightweight TTS microservice for the Grand Egyptian Museum. "
        "Converts artifact descriptions to speech using ElevenLabs. "
        "Caching is handled by the Node.js backend."
    ),
    version="2.0.0",
    lifespan=lifespan,
    # Disable interactive docs in production
    docs_url="/docs" if os.getenv("ENV") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENV") != "production" else None,
)

# ── Middleware ─────────────────────────────────────────────────────────────

# Request ID — must be registered first so all downstream middleware
# and handlers can access request.state.request_id
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

# ── Static Files ───────────────────────────────────────────────────────────
# Serves generated MP3 files at /static/audio/
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Router Registration ────────────────────────────────────────────────────
app.include_router(generate_router)


# ── Health Check ──────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "voice_tour_guide",
        "port": 8003,
        "version": "2.0.0",
    }


# ── Graceful Shutdown ─────────────────────────────────────────────────────
def _handle_shutdown(signum, frame):
    logger.info("Shutdown signal received", extra={"signal": signum})
    sys.exit(0)


signal.signal(signal.SIGTERM, _handle_shutdown)
signal.signal(signal.SIGINT, _handle_shutdown)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8003,
        reload=False,
        access_log=False,  # Logging is handled via structured middleware
    )
