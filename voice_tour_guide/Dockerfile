# ── Base Image ────────────────────────────────────────────────────────────────
FROM python:3.12-slim

# ── System Dependencies ───────────────────────────────────────────────────────
# libsndfile1 : audio file I/O (used by audio processing utilities)
# ffmpeg      : audio format conversion
# curl        : used by the Docker health check
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Non-root User ─────────────────────────────────────────────────────────────
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# ── Working Directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python Dependencies ───────────────────────────────────────────────────────
# Copy requirements first to leverage Docker layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Code ──────────────────────────────────────────────────────────
COPY . .

# ── Audio Output Directory ────────────────────────────────────────────────────
# Created at build time so the directory exists before the first request arrives.
# chown ensures the non-root user can write generated MP3 files here.
RUN mkdir -p static/audio && chown -R appuser:appgroup /app

# ── Switch to Non-root User ────────────────────────────────────────────────────
USER appuser

# ── Service Port ──────────────────────────────────────────────────────────────
# Port assignments: 8000=cv-recognition | 8001=chatbot-llm |
#                  8002=hieroglyph-translator | 8003=voice-tour-guide
EXPOSE 8003

# ── Health Check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8003/health || exit 1

# ── Startup Command ───────────────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", \
     "--no-access-log"]
