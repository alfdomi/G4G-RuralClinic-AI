#!/bin/bash
set -e

# ──────────────────────────────────────────────────────────────────────────────
# Docker entrypoint for G4G RuralClinic AI (App Service only)
#
# This container runs the Python/Gradio application.
# It expects Ollama to be running at OLLAMA_HOST (default: http://ollama:11434).
#
# Environment variables (see src/config.py):
#   OLLAMA_HOST, OFFLINE_MODE, USE_QUANTIZED, QUANTIZED_MODEL_NAME, LOG_LEVEL
#
# Docker Compose mounts (recommended):
#   - ./ollama_data:/root/.ollama   (on ollama service)
#   - ./data:/app/data              (shared data)
#   - ./logs:/app/logs              (shared logs)
#
# Usage:
#   docker run -p 7860:7860 \
#     -e OLLAMA_HOST=http://ollama-host:11434 \
#     -v ./data:/app/data \
#     -v ./logs:/app/logs \
#     rural-clinic-ai
#
# ──────────────────────────────────────────────────────────────────────────────

export OLLAMA_HOST=${OLLAMA_HOST:-http://ollama:11434}
export OFFLINE_MODE=${OFFLINE_MODE:-true}
export USE_QUANTIZED=${USE_QUANTIZED:-true}
export QUANTIZED_MODEL_NAME=${QUANTIZED_MODEL_NAME:-gemma4:e4b-q4_K_M}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

echo "========================================"
echo " G4G RuralClinic AI — Application"
echo "========================================"
echo "Ollama host:   $OLLAMA_HOST"
echo "Offline mode:  $OFFLINE_MODE"
echo "Model:         $QUANTIZED_MODEL_NAME"
echo "========================================"
echo ""

# Quick health check: can we reach Ollama?
if curl -s -f "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
    echo "✓ Ollama is reachable."
    echo ""
else
    echo "⚠️  Warning: Cannot reach Ollama at $OLLAMA_HOST"
    echo "   Make sure Ollama is running and accessible."
    echo "   The app will start but may fail to serve requests."
    echo ""
fi

# Launch the Gradio app
exec python -m src.app.demo
