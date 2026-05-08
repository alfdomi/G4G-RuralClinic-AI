# ────────────────────────────────────────────────────────────────────────────────
# Dockerfile for G4G RuralClinic AI
# 
# Multi-stage build for a small, efficient final image.
# This image contains the Python application (Gradio UI + orchestrator/workers).
# Ollama runs as a separate service (see docker-compose.yml).
#
# Build args:
#   - PYTHON_VERSION: Python base image version
#
# Usage:
#   docker build -t rural-clinic-ai .
#   docker compose up --build   (recommended, uses docker-compose.yml)
#
# Notes:
#   - Uses multi-stage build to keep final image small (~400 MB vs ~1 GB).
#   - data/ and logs/ are copied into the image at build time so that
#     the app can run standalone, but in production you should mount
#     volumes to persist data across container restarts.
#
# ────────────────────────────────────────────────────────────────────────────────

# ── Stage 1: Builder — install deps and build the app ───────────────────────────
FROM python:3.12-slim AS app-builder

ENV DEBIAN_FRONTEND=noninteractive

# System deps required at build time for compiling native packages
# (Pillow, numpy, opencv-python-headless, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer — rebuilds only when requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/     ./src/
COPY eval/    ./eval/
COPY scripts/ ./scripts/

# Copy data directory if it exists and has content.
# Docker COPY fails if the source directory doesn't exist or is empty on some platforms.
# We use a .dockerignore-friendly approach: copy everything that exists, ignoring errors.
# The data directory is also created below as a fallback.
COPY data/ ./data/

# ── Stage 2: Runtime — minimal image with no build tools ────────────────────────
FROM python:3.12-slim AS runtime

# Avoid interactive prompts and install only runtime libs
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder (avoids re-downloading)
COPY --from=app-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=app-builder /usr/local/bin /usr/local/bin

# Copy all application code from builder
WORKDIR /app
COPY --from=app-builder --chmod=755 /app /app

# Ensure data and logs directories exist with proper permissions
# (data/ may have been empty at build time, and we always need logs/)
RUN mkdir -p /app/data /app/logs \
    && chmod 755 /app/data /app/logs

# ── Runtime configuration ───────────────────────────────────────────────────────
# These defaults can be overridden via docker run -e or .env file.
# See .env.example for recommended settings.
ENV OLLAMA_HOST=http://ollama:11434
ENV OFFLINE_MODE=true
ENV USE_QUANTIZED=true
ENV QUANTIZED_MODEL_NAME=gemma4:e4b-q4_K_M
ENV LOG_LEVEL=INFO

# Expose the Gradio web UI port
EXPOSE 7860

# Launch the Gradio application (docker-entrypoint.sh or CMD override)
CMD ["python", "-m", "src.app.demo"]

# ────────────────────────────────────────────────────────────────────────────────