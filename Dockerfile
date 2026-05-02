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
#
# ────────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim AS app-builder

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps for Pillow, OpenCV, and numerical Python
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

# Create app directory
WORKDIR /app

# Copy and install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ ./src/
COPY eval/ ./eval/
COPY tests/ ./tests/

# Copy data and scripts (sample images, knowledge base, scripts)
COPY data/ ./data/ 2>/dev/null || true
COPY scripts/ ./scripts/

# Create run directory and set permissions
RUN mkdir -p /app/data /app/logs


# ── Runtime stage (same image, no build tools needed) ─────────────────────────
FROM python:3.12-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libopenblas0 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python environment from builder
COPY --from=app-builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=app-builder /usr/local/bin /usr/local/bin

# Copy application code
WORKDIR /app
COPY --from=app-builder /app /app

# Runtime configuration (can be overridden via docker run -e)
# See also: .env.example
ENV OLLAMA_HOST=http://ollama:11434
ENV OFFLINE_MODE=true
ENV USE_QUANTIZED=true
ENV QUANTIZED_MODEL_NAME=gemma4:e4b-q4_K_M
ENV LOG_LEVEL=INFO

# Create runtime directories
RUN mkdir -p /app/data /app/logs

EXPOSE 7860

# Launch the Gradio application
CMD ["python", "-m", "src.app.demo"]

# ────────────────────────────────────────────────────────────────────────────────
