FROM python:3.11-slim

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

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer cache:
# the heavy pip install layer is only re-run when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY . .

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8001

# ── Health check ──────────────────────────────────────────────────────────────
# Docker / docker-compose will mark the container as healthy only after the
# /health endpoint returns HTTP 200 (model loading can take 30-60 s on cold start).
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
