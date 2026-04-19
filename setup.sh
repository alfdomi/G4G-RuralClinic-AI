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

if command -v ollama &> /dev/null; then
    # Try E4B (4B params) first, then E2B (2B) for lower-RAM machines
    if ollama pull gemma4:e4b 2>/dev/null; then
        echo "      Pulled: gemma4:e4b"
    elif ollama pull gemma4:e2b 2>/dev/null; then
        echo "      Pulled: gemma4:e2b (lighter variant, targets <8 GB RAM)"
    else
        echo "      WARNING: Could not pull Gemma 4 via Ollama."
        echo "      Try manually: ollama pull gemma4:e4b"
        echo "      Or use the HuggingFace fallback (see src/config.py)."
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
echo "OFFLINE MODE (no internet after model pull):"
echo "  - Set OFFLINE_MODE=true in src/config.py"
echo "  - Or export OFFLINE_MODE=true before launching"
echo ""
echo "DOWNLOAD SAMPLE IMAGES (optional):"
echo "  bash scripts/download_sample_data.sh"
echo ""
echo "RUN TESTS:"
echo "  python -m pytest tests/ -v"
echo ""
