FROM python:3.11-slim

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ───────────────────────────────────────────────────────
# curl is required by the HEALTHCHECK instruction below.
# libgl1 + libglib2.0-0 are required by OpenCV (pulled in by ultralytics).
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
# model/best_V2.pt is large and is mounted as a bind volume at runtime,
# so it does NOT need to be baked into the image. See .dockerignore.
COPY . .

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8002

# ── Health check ──────────────────────────────────────────────────────────────
# 90s start_period allows YOLO model to load before health checks begin.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]
