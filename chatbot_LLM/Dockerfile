FROM python:3.12-slim

# ── Environment ───────────────────────────────────────────────────────────────
# HF_HOME + SENTENCE_TRANSFORMERS_HOME: point both caches to the mounted volume
# so model weights survive container restarts without re-downloading.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/models/huggingface

# ── System dependencies ───────────────────────────────────────────────────────
# build-essential  → needed to compile some Python C extensions (e.g. hnswlib inside chromadb)
# curl             → used by the HEALTHCHECK below
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root User ─────────────────────────────────────────────────────────────
# UID/GID 1000 intentionally matches the host user (first non-root user on Linux).
# This is required so that bind-mounted directories (e.g. models/vectordb) are
# writable by appuser inside the container — Docker bind-mounts bypass image-layer
# ownership, so the in-container UID must equal the host file owner UID.
RUN groupadd --gid 1000 appgroup \
    && useradd --uid 1000 --gid 1000 --no-create-home --shell /sbin/nologin appuser

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer cache:
# the heavy pip install layer is only re-run when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# ── Ensure runtime directories exist and have correct ownership ───────────────
# static/audio: FastAPI StaticFiles mount fails if absent.
# models/vectordb: ChromaDB bind-mount must be group-writable for appuser.
RUN mkdir -p static/audio models/vectordb \
    && chown -R appuser:appgroup /app \
    && chmod -R g+w models/vectordb

# ── Switch to Non-root User ────────────────────────────────────────────────────
USER appuser

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8001

# ── Health check ──────────────────────────────────────────────────────────────
# Docker / docker-compose will mark the container as healthy only after the
# /health endpoint returns HTTP 200 (model loading can take 30-60 s on cold start).
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--no-access-log"]
