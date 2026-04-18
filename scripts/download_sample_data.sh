#!/usr/bin/env bash
# Download sample dermatology images for testing the G4G RuralClinic AI system.
#
# Primary source: DermaMNIST via HuggingFace Datasets (10 test samples).
# Fallback: generates 5 synthetic colour-patch images using Pillow + NumPy.
#
# Images are saved to data/samples/ — this directory is gitignored.
# Run this script once after setup to have images ready for smoke-testing.
#
# Usage:
#   bash scripts/download_sample_data.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SAMPLE_DIR="$ROOT_DIR/data/samples"

echo "================================================"
echo " G4G RuralClinic AI — Sample Data Download"
echo "================================================"
echo "Target directory: $SAMPLE_DIR"
echo ""

mkdir -p "$SAMPLE_DIR"

# Export for use inside the Python heredoc
export SAMPLE_DIR

python3 - <<'PYTHON'
import os
import sys
from pathlib import Path

sample_dir = Path(os.environ["SAMPLE_DIR"])
sample_dir.mkdir(parents=True, exist_ok=True)

# ── Attempt 1: DermaMNIST via HuggingFace Datasets ───────────────────────────
try:
    from datasets import load_dataset

    print("Attempting to load DermaMNIST from HuggingFace (10 test samples)…")
    dataset = load_dataset(
        "albertvillanova/medmnist-v2", "dermamnist", split="test[:10]"
    )
    for i, sample in enumerate(dataset):
        img = sample["image"]
        out = sample_dir / f"dermamnist_{i:03d}_label{sample['label']}.jpg"
        img.convert("RGB").save(out, quality=95)

    print(f"Saved {len(dataset)} DermaMNIST samples to {sample_dir}")
    sys.exit(0)

except ImportError:
    print("'datasets' package not available. Run: pip install datasets")
except Exception as exc:
    print(f"DermaMNIST download failed: {exc}")

# ── Attempt 2: Synthetic test images (always works offline) ──────────────────
print("\nGenerating synthetic test images as fallback…")

try:
    import numpy as np
    from PIL import Image, ImageDraw

    # Each tuple: (base_colour_RGB, label, description)
    samples = [
        ((80,  50,  30),  "dark_irregular",   "Simulates a dark, irregular lesion"),
        ((180, 120, 90),  "medium_brown_mole", "Simulates a medium-brown mole"),
        ((200, 160, 140), "light_patch",       "Simulates a light skin patch"),
        ((160, 60,  50),  "reddish_spot",      "Simulates a reddish inflamed spot"),
        ((210, 190, 170), "normal_skin",       "Simulates normal surrounding skin"),
    ]

    for colour, name, desc in samples:
        arr = np.full((224, 224, 3), colour, dtype=np.uint8)
        # Add mild Gaussian-like noise for realism
        noise = np.random.randint(-15, 15, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        img = Image.fromarray(arr)
        draw = ImageDraw.Draw(img)
        # Draw a rough lesion shape in the centre
        cx, cy, r = 112, 112, 40
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour)

        out = sample_dir / f"synthetic_{name}.jpg"
        img.save(out, quality=95)
        print(f"  Created: {out.name}  ({desc})")

    print(f"\nCreated {len(samples)} synthetic images in {sample_dir}")

except Exception as exc:
    print(f"Synthetic image generation failed: {exc}")
    sys.exit(1)
PYTHON

echo ""
echo "================================================"
echo " Done! Sample images are in:"
echo "   $SAMPLE_DIR"
echo ""
echo " Run tests with:"
echo "   python -m pytest tests/ -v"
echo ""
echo " Launch the demo with:"
echo "   python -m src.app.demo"
echo "================================================"
