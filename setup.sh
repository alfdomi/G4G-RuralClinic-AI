#!/usr/bin/env bash
# Setup script for G4G RuralClinic AI
# Installs dependencies and pulls the Gemma 4 model via Ollama for offline use.

set -e

echo "============================================"
echo " G4G RuralClinic AI — Setup"
echo "============================================"
echo ""

# Step 1: Install Python dependencies
echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt
echo "      Done."
echo ""

# Step 2: Pull Gemma 4 model via Ollama
echo "[2/3] Pulling Gemma 4 model via Ollama..."
echo "      This requires Ollama to be installed: https://ollama.com"
echo ""

# RAM check (approximate)
TOTAL_RAM_KB=$(grep MemTotal /proc/meminfo 2>/dev/null | awk '{print $2}')
TOTAL_RAM_GB=$((TOTAL_RAM_KB / 1024 / 1024))
if [ -z "$TOTAL_RAM_GB" ] || [ "$TOTAL_RAM_GB" -eq 0 ] 2>/dev/null; then
    TOTAL_RAM_GB="unknown"
fi
echo "      System RAM detected: ~${TOTAL_RAM_GB} GB"
echo ""

# Model selection based on available RAM and user preference
# Default quantized model: gemma4:e4b-q4_K_M (4-bit, ~4 GB RAM)
USE_QUANTIZED="${USE_QUANTIZED:-true}"
QUANTIZED_MODEL="${QUANTIZED_MODEL_NAME:-gemma4:e4b-q4_K_M}"

echo "      Quantized enabled: $USE_QUANTIZED"
echo "      Quantized model:   $QUANTIZED_MODEL"
echo ""

if command -v ollama &> /dev/null; then
    if [ "$USE_QUANTIZED" = "true" ] || [ "$USE_QUANTIZED" = "1" ]; then
        echo "      Attempting quantized model..."
        # Try user-specified quantized model, fall back to q4_K_M, then e4b, then e2b
        for m in "$QUANTIZED_MODEL" "gemma4:e4b-q4_K_M" "gemma4:e4b-q8_0" "gemma4:e4b" "gemma4:e2b"; do
            echo "      Pulling $m ..."
            if ollama pull "$m" 2>/dev/null; then
                echo "      ✓ Successfully pulled: $m"
                echo ""
                echo "      RAM estimate: $(grep -A1 "$m" <<< '
                gemma4:e4b-q4_K_M 4
                gemma4:e4b-q8_0 6
                gemma4:e4b 6
                gemma4:e2b 3' 2>/dev/null | tail -1 | awk '{print $2}') GB (approx)"
                echo "      Set USE_QUANTIZED=false to use full precision."
                break
            fi
        done
    else
        echo "      Attempting full precision model..."
        if ollama pull gemma4:e4b 2>/dev/null; then
            echo "      ✓ Successfully pulled: gemma4:e4b"
        elif ollama pull gemma4:e2b 2>/dev/null; then
            echo "      ✓ Successfully pulled: gemma4:e2b"
        fi
    fi
else
    echo "      WARNING: Ollama not found. Install from https://ollama.com"
    echo "      The app will fall back to the HuggingFace transformers pipeline."
fi
echo ""

# Step 3: Create required directories
echo "[3/3] Creating project directories..."
mkdir -p data/samples logs
echo "      Created: data/ logs/"
echo ""

# Done
echo "============================================"
echo " Setup complete!"
echo "============================================"
echo ""
echo "OFFLINE USAGE INSTRUCTIONS:"
echo "  1. Start Ollama:       ollama serve"
echo "  2. Launch the demo:    python -m src.app.demo"
echo "  3. Open browser:       http://localhost:7860"
echo ""
echo "QUANTIZED MODELS (recommended for low-RAM systems):"
echo "  gemma4:e4b-q4_K_M    ~4 GB RAM  — best balance (DEFAULT)"
echo "  gemma4:e4b-q8_0      ~6 GB RAM  — higher quality 8-bit"
echo "  gemma4:e2b-q4_K_M    ~2 GB RAM  — ultra-lightweight"
echo "  gemma4:e2b-q8_0      ~3 GB RAM  — lightweight 8-bit"
echo ""
echo "ENVIRONMENT VARIABLES:"
echo "  USE_QUANTIZED=true           — enable quantized model (default: true)"
echo "  QUANTIZED_MODEL_NAME=...     — choose specific quantized variant"
echo "  OFFLINE_MODE=true            — disable all network calls"
echo "  OLLAMA_HOST=http://...       — set Ollama endpoint"
echo ""
echo "DOWNLOAD SAMPLE IMAGES (optional):"
echo "  bash scripts/download_sample_data.sh"
echo ""
echo "RUN TESTS:"
echo "  python -m pytest tests/ -v"
echo ""
echo "DOCKER USAGE (see docker-compose.yml):"
echo "  docker compose up --build   — build + launch with CPU-only mode"
echo ""
