FROM python:3.12-slim

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── System dependencies ───────────────────────────────────────────────────────
# curl is required by the HEALTHCHECK instruction below.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root User ─────────────────────────────────────────────────────────────
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer cache.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application source ────────────────────────────────────────────────────────
# model/model.h5 is large (216 MB) and is mounted as a bind volume at runtime,
# so it does NOT need to be baked into the image. See .dockerignore.
COPY . .

# ── Ownership & Switch to Non-root User ────────────────────────────────────
RUN chown -R appuser:appgroup /app
USER appuser

# ── Port ──────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ── Health check ──────────────────────────────────────────────────────────────
# 60s start_period allows TensorFlow and the model to load before health checks begin.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entry point ───────────────────────────────────────────────────────────────
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
