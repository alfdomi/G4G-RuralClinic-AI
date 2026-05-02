"""
Configuration for G4G RuralClinic AI.

Central settings for Gemma 4 model, offline mode, image processing,
paths, and safety constants. Supports environment variable overrides.

Project: Offline multimodal AI dermatology assistant powered by Gemma 4.
Target hardware: Standard laptop, 8–16 GB RAM, CPU-first inference.
"""

import os
from pathlib import Path

# ─── Quantization Support ─────────────────────────────────────────────────────
# Quantized models (GGUF) for efficient CPU inference on low-end hardware.
# Ollama handles quantization via :q4_K_M, :q8_0, etc. suffixes.
# RAM estimates include model weights + context overhead for CPU inference.
QUANTIZED_MODELS = {
    "gemma4:e4b-q8_0": {
        "ram_gb": 6.0,
        "description": "E4B 8-bit quantized — best quality/speed balance for 8+ GB RAM",
        "quality": "high",
        "recommended_for": "laptops with 8–16 GB RAM",
        "ollama_tag": "gemma4:e4b-q8_0",
    },
    "gemma4:e4b-q4_K_M": {
        "ram_gb": 4.0,
        "description": "E4B 4-bit K-quant — excellent for 4–8 GB RAM, minimal quality loss",
        "quality": "medium-high",
        "recommended_for": "laptops with 4–8 GB RAM",
        "ollama_tag": "gemma4:e4b-q4_K_M",
    },
    "gemma4:e4b-q4_0": {
        "ram_gb": 3.8,
        "description": "E4B 4-bit quantized — lighter alternative, good for constrained hardware",
        "quality": "medium",
        "recommended_for": "laptops with 4–6 GB RAM",
        "ollama_tag": "gemma4:e4b-q4_0",
    },
    "gemma4:e2b-q8_0": {
        "ram_gb": 3.0,
        "description": "E2B 8-bit quantized — lightweight, fast inference",
        "quality": "medium",
        "recommended_for": "laptops with 2–4 GB RAM",
        "ollama_tag": "gemma4:e2b-q8_0",
    },
    "gemma4:e2b-q4_K_M": {
        "ram_gb": 2.0,
        "description": "E2B 4-bit K-quant — ultra-lightweight for constrained hardware",
        "quality": "medium",
        "recommended_for": "laptops with 2–4 GB RAM",
        "ollama_tag": "gemma4:e2b-q4_K_M",
    },
}

# Full precision / high-bit models (reference)
FULL_PRECISION_MODELS = {
    "gemma4:e4b": {
        "ram_gb": 6.0,
        "description": "E4B (Ollama default ~4-bit)",
        "quality": "high",
        "recommended_for": "laptops with 8+ GB RAM",
        "ollama_tag": "gemma4:e4b",
    },
    "gemma4:e2b": {
        "ram_gb": 3.0,
        "description": "E2B (Ollama default ~4-bit)",
        "quality": "medium",
        "recommended_for": "laptops with <8 GB RAM",
        "ollama_tag": "gemma4:e2b",
    },
}

# ─── Quantization / Model Selection ──────────────────────────────────────────
# Set USE_QUANTIZED=true (env var) to enable quantized model selection.
# QUANTIZED_MODEL_NAME specifies which quantized variant to use.
USE_QUANTIZED: bool = os.environ.get("USE_QUANTIZED", "true").lower() in ("1", "true", "yes")
QUANTIZED_MODEL_NAME: str = os.environ.get("QUANTIZED_MODEL_NAME", "gemma4:e4b-q4_K_M")

# ─── Model Configuration ─────────────────────────────────────────────────────
# Gemma 4 E4B instruction-tuned variant via Ollama (CPU-friendly, ~4B params)
# Pull command: ollama pull gemma4:e4b
# Low-end hardware alternative: ollama pull gemma4:e2b
# Quantized variants: ollama pull gemma4:e4b-q4_K_M  (recommended for 4-8 GB RAM)
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

# ─── RAG (Retrieval-Augmented Generation) ─────────────────────────────────────
RAG_TOP_K: int = 3             # Knowledge chunks returned per query
RAG_KNOWLEDGE_BASE_PATH = ROOT_DIR / "src" / "rag" / "knowledge_base.json"

# Comprehensive guidelines KB: 45 chunks covering ABCDE, all skin classes,
# skin-of-color, Fitzpatrick types, rural/tropical conditions, WHO referral.
DERM_GUIDELINES_PATH = ROOT_DIR / "data" / "knowledge" / "derm_guidelines.json"

# ─── Effective Model Selection ────────────────────────────────────────────────
# Determine which model name to use based on quantization setting.
# This is computed at import time from environment variables.
def get_effective_model_name() -> str:
    """
    Return the model name (Ollama tag) to use based on USE_QUANTIZED setting.
    Respects env vars: USE_QUANTIZED, QUANTIZED_MODEL_NAME.
    """
    if USE_QUANTIZED:
        m = QUANTIZED_MODEL_NAME
        # Validate it's a known quantized model; fall back to default if not.
        if m in QUANTIZED_MODELS:
            return m
        else:
            return "gemma4:e4b-q4_K_M"  # safest default
    else:
        return MODEL_NAME  # gemma4:e4b (Ollama default ~4-bit)

EFFECTIVE_MODEL_NAME: str = get_effective_model_name()

# Estimated RAM requirement for the currently selected effective model
def get_model_ram_estimate(model_name: str) -> float:
    """Return RAM estimate in GB for a given model name."""
    if model_name in QUANTIZED_MODELS:
        return QUANTIZED_MODELS[model_name]["ram_gb"]
    if model_name in FULL_PRECISION_MODELS:
        return FULL_PRECISION_MODELS[model_name]["ram_gb"]
    return 6.0  # safe default
