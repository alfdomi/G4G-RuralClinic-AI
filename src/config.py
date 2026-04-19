"""
Configuration for G4G RuralClinic AI.

Central settings for Gemma 4 model, offline mode, image processing,
paths, and safety constants. Supports environment variable overrides.

Project: Offline multimodal AI dermatology assistant powered by Gemma 4.
Target hardware: Standard laptop, 8–16 GB RAM, CPU-first inference.
"""

import os
from pathlib import Path

# ─── Model Configuration ─────────────────────────────────────────────────────
# Gemma 4 E4B instruction-tuned variant via Ollama (CPU-friendly, ~4B params)
# Pull command: ollama pull gemma4:e4b
# Low-end hardware alternative: ollama pull gemma4:e2b
MODEL_NAME = "gemma4:e4b"
HF_MODEL_NAME = "google/gemma-4-e4b-it"   # HuggingFace fallback (image-text-to-text)

# ─── Offline Mode ─────────────────────────────────────────────────────────────
# True = disable all internet calls after initial model pull.
# Override with: export OFFLINE_MODE=false
OFFLINE_MODE: bool = os.environ.get("OFFLINE_MODE", "true").lower() in ("1", "true", "yes")

# ─── Image Processing ─────────────────────────────────────────────────────────
IMAGE_SIZE = (224, 224)        # Standard resize target for lesion images
MAX_IMAGE_SIZE_MB = 10         # Reject inputs above this threshold

# ─── Project Paths ────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
LOGS_DIR = ROOT_DIR / "logs"
SAMPLE_DATA_DIR = DATA_DIR / "samples"

DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# ─── Safety Disclaimer ────────────────────────────────────────────────────────
# This constant is appended to ALL AI outputs. Never remove or shorten it.
SAFETY_DISCLAIMER = (
    "IMPORTANT: I am not a doctor. This AI-generated analysis is for "
    "educational and informational purposes only. It is NOT a medical diagnosis. "
    "Always consult a qualified dermatologist or healthcare professional for any "
    "skin concerns, especially if you notice changes in moles or skin lesions. "
    "Early professional evaluation can be life-saving."
)

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "dermatology_ai.log"

# ─── Ollama Backend ───────────────────────────────────────────────────────────
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_TIMEOUT: int = 120   # CPU inference with an image takes time

# ─── Conversation Memory ──────────────────────────────────────────────────────
MAX_HISTORY_LENGTH: int = 10   # Max user/assistant turns retained in memory
