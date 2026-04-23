"""
Evaluation harness for G4G RuralClinic AI — DermatologyWorker.

Supports two data source modes:
  --data-dir  Real images organised in class-named subdirectories
              (ISIC / HAM10000 directory-tree format):
                  data/samples/melanoma/*.jpg
                  data/samples/benign_nevus/*.jpg
                  ...
  --mock      Synthetic benchmark: generates solid-colour test images and
              patches DermatologyWorker.analyze() to return pre-defined
              responses, so the evaluation runs without Ollama.

Metrics produced:
  - Overall accuracy
  - Per-class precision, recall, F1
  - Macro-average F1
  - Confusion matrix (as a dict of dicts)

Output is written to JSON (default: logs/eval_results.json) and printed
as a summary table to stdout.

Usage examples:
    # Dry run with synthetic data (no Ollama required):
    python -m eval.evaluate --mock

    # Real evaluation on downloaded ISIC samples:
    python -m eval.evaluate --data-dir data/samples --max-samples 100

    # ISIC CSV format (flat image dir + ground-truth CSV):
    python -m eval.evaluate --data-dir data/isic --csv data/isic/labels.csv

Project: G4G RuralClinic AI — offline frontier AI for accessible
dermatology screening in rural and underserved communities.
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from PIL import Image

# Ensure project root is importable when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workers.dermatology import SKIN_CLASSES, DermatologyWorker
from src.config import LOGS_DIR, SAMPLE_DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ─── Mock response fixtures ───────────────────────────────────────────────────

def _mock_response(classification: str) -> dict:
    """Pre-canned correct response for synthetic evaluation."""
    return {
        "classification": classification,
        "confidence": "medium",
        "visual_description": f"Synthetic test image for class {classification}.",
        "risk_level": "low" if classification in {"benign_nevus", "normal_skin", "seborrheic_keratosis", "dermatofibroma"} else "moderate",
        "plain_explanation": f"This is a synthetic test sample labelled {classification}.",
        "recommended_action": "Consult a dermatologist.",
        "abcde_notes": "Synthetic — not applicable.",
        "disclaimer": "Test mode — not a real analysis.",
        "raw_model_output": "{}",
    }


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_directory_tree(
    data_dir: Path,
    max_per_class: Optional[int] = None,
) -> list[tuple[Path, str]]:
    """
    Load (image_path, label) pairs from a directory tree:
        data_dir/
            melanoma/img1.jpg  img2.jpg ...
            benign_nevus/img1.jpg ...

    Only directories whose names appear in SKIN_CLASSES are used.
    Returns a list of (Path, label) tuples.
    """
    samples: list[tuple[Path, str]] = []
    for cls in SKIN_CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            continue
        images = sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")) + sorted(cls_dir.glob("*.jpeg"))
        if max_per_class:
            images = images[:max_per_class]
        for img_path in images:
            samples.append((img_path, cls))
        if images:
            logger.info("  %s: %d images", cls, len(images))
    return samples


def load_isic_csv(
    data_dir: Path,
    csv_path: Path,
    max_samples: Optional[int] = None,
) -> list[tuple[Path, str]]:
    """
    Load (image_path, label) from an ISIC-style CSV with columns:
        image,melanoma,basal_cell_carcinoma,...  (one-hot)
    or:
        image,label  (string label)

    Falls back gracefully if CSV columns don't match SKIN_CLASSES.
    """
    import csv

    samples: list[tuple[Path, str]] = []

    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []

        # Detect format: one-hot vs. string label
        one_hot_cols = [f for f in fieldnames if f in SKIN_CLASSES]
        has_label_col = "label" in fieldnames or "dx" in fieldnames
        label_col = "label" if "label" in fieldnames else "dx"

        for row in reader:
            img_name = row.get("image") or row.get("image_id") or ""
            if not img_name:
                continue

            # Try common extensions
            img_path = None
            for ext in (".jpg", ".jpeg", ".png", ""):
                candidate = data_dir / (img_name + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            if one_hot_cols:
                # Pick the column with value "1" (or highest float)
                best = max(one_hot_cols, key=lambda c: float(row.get(c, 0) or 0))
                label = best
            elif has_label_col:
                label = row.get(label_col, "unknown").lower().replace(" ", "_")
                if label not in SKIN_CLASSES:
                    label = "unknown"
            else:
                continue

            samples.append((img_path, label))
            if max_samples and len(samples) >= max_samples:
                break

    logger.info("Loaded %d samples from CSV %s", len(samples), csv_path)
    return samples


def build_synthetic_samples(n_per_class: int = 3) -> list[tuple[Image.Image, str]]:
    """
    Generate solid-colour 224×224 PIL images as synthetic test data.
    One distinct colour per class, n_per_class images each.
    Returns (PIL Image, label) tuples (no file paths needed for mock mode).
    """
    import numpy as np

    colours = [
        (180, 100, 100), (100, 180, 100), (100, 100, 180),
        (180, 180, 100), (100, 180, 180), (180, 100, 180),
        (150, 120, 90), (90, 150, 120), (120, 90, 150), (160, 160, 160),
    ]
    samples = []
    for cls, colour in zip(SKIN_CLASSES, colours):
        for _ in range(n_per_class):
            arr = np.full((224, 224, 3), colour, dtype=np.uint8)
            samples.append((Image.fromarray(arr), cls))
    return samples


# ─── Metrics ─────────────────────────────────────────────────────────────────

def compute_metrics(
    predictions: list[str],
    ground_truth: list[str],
    classes: list[str],
) -> dict:
    """
    Compute accuracy, per-class P/R/F1, macro-F1, and confusion matrix.

    Returns a dict suitable for JSON serialisation.
    """
    n = len(predictions)
    if n == 0:
        return {"error": "No predictions to evaluate"}

    # Accuracy
    accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / n

    # Per-class metrics
    per_class: dict[str, dict] = {}
    f1_scores: list[float] = []
    for cls in classes:
        tp = sum(p == cls and g == cls for p, g in zip(predictions, ground_truth))
        fp = sum(p == cls and g != cls for p, g in zip(predictions, ground_truth))
        fn = sum(p != cls and g == cls for p, g in zip(predictions, ground_truth))
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
        }
        if support > 0:
            f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    # Confusion matrix
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pred, gt in zip(predictions, ground_truth):
        confusion[gt][pred] += 1
    confusion_dict = {k: dict(v) for k, v in confusion.items()}

    return {
        "n_samples": n,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
        "confusion_matrix": confusion_dict,
    }


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_evaluation(
    samples,
    worker: DermatologyWorker,
    mock: bool = False,
    mock_correct_rate: float = 0.70,
) -> dict:
    """
    Run inference on all samples and collect predictions.

    Args:
        samples:   list of (image_or_path, label) pairs.
        worker:    DermatologyWorker instance.
        mock:      If True, inject pre-defined responses instead of running inference.
        mock_correct_rate: Fraction of mock predictions that match ground truth
                           (remainder are assigned to "unknown") — simulates realistic
                           partial accuracy without needing Ollama.

    Returns:
        Metrics dict from compute_metrics().
    """
    import random
    random.seed(42)

    predictions: list[str] = []
    ground_truth: list[str] = []
    errors = 0

    for i, (img_or_path, label) in enumerate(samples):
        # Resolve image
        if isinstance(img_or_path, Path):
            try:
                image = Image.open(img_or_path).convert("RGB")
            except Exception as exc:
                logger.warning("Could not open %s: %s", img_or_path, exc)
                errors += 1
                continue
        else:
            image = img_or_path

        # Run analysis
        if mock:
            # Inject a correct or incorrect response based on mock_correct_rate
            if random.random() < mock_correct_rate:
                result = _mock_response(label)
            else:
                alt = random.choice([c for c in SKIN_CLASSES if c != label])
                result = _mock_response(alt)
        else:
            try:
                result = worker.analyze(image=image, symptoms="")
            except Exception as exc:
                logger.warning("Worker error on sample %d (%s): %s", i, label, exc)
                result = {"classification": "unknown"}
                errors += 1

        pred = result.get("classification", "unknown")
        predictions.append(pred)
        ground_truth.append(label)

        if (i + 1) % 10 == 0:
            logger.info("  Progress: %d/%d samples processed", i + 1, len(samples))

    if errors:
        logger.warning("%d samples skipped due to errors", errors)

    metrics = compute_metrics(predictions, ground_truth, SKIN_CLASSES)
    metrics["errors_skipped"] = errors
    return metrics


def print_report(metrics: dict) -> None:
    """Print a formatted evaluation report to stdout."""
    print("\n" + "=" * 60)
    print("  G4G RuralClinic AI — Evaluation Report")
    print("=" * 60)
    print(f"  Samples evaluated : {metrics.get('n_samples', 0)}")
    print(f"  Accuracy          : {metrics.get('accuracy', 0):.1%}")
    print(f"  Macro F1          : {metrics.get('macro_f1', 0):.1%}")
    if metrics.get("errors_skipped"):
        print(f"  Errors skipped    : {metrics['errors_skipped']}")
    print()
    print(f"  {'Class':<30} {'P':>6} {'R':>6} {'F1':>6} {'N':>5}")
    print("  " + "-" * 55)
    for cls, m in (metrics.get("per_class") or {}).items():
        if m["support"] > 0:
            print(
                f"  {cls:<30} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                f"{m['f1']:>6.3f} {m['support']:>5}"
            )
    print("=" * 60 + "\n")


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate G4G RuralClinic AI DermatologyWorker"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SAMPLE_DATA_DIR,
        help="Directory with class-named subdirs (default: data/samples)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional ISIC-style CSV with image IDs and labels",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=200,
        help="Maximum total samples to evaluate (default: 200)",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        help="Maximum samples per class when using directory tree",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic images and mock inference — no Ollama required",
    )
    parser.add_argument(
        "--mock-accuracy",
        type=float,
        default=0.70,
        help="Fraction of correct mock predictions (0–1, default: 0.70)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LOGS_DIR / "eval_results.json",
        help="JSON output path (default: logs/eval_results.json)",
    )
    args = parser.parse_args()

    # ── Load samples ──────────────────────────────────────────────────────
    if args.mock:
        logger.info("Mock mode: generating synthetic samples")
        n_per_class = max(1, args.max_samples // len(SKIN_CLASSES))
        samples = build_synthetic_samples(n_per_class=n_per_class)
        logger.info("Generated %d synthetic samples across %d classes", len(samples), len(SKIN_CLASSES))

    elif args.csv and args.csv.exists():
        logger.info("Loading samples from CSV: %s", args.csv)
        samples = load_isic_csv(args.data_dir, args.csv, max_samples=args.max_samples)

    elif args.data_dir.is_dir():
        logger.info("Loading samples from directory tree: %s", args.data_dir)
        max_per = args.max_per_class or (args.max_samples // len(SKIN_CLASSES) or None)
        samples = load_directory_tree(args.data_dir, max_per_class=max_per)
        if args.max_samples:
            samples = samples[:args.max_samples]
        logger.info("Loaded %d samples total", len(samples))

    else:
        logger.warning(
            "No data found at %s. Use --mock for a synthetic dry run, "
            "or download sample data with: bash scripts/download_sample_data.sh",
            args.data_dir,
        )
        parser.print_help()
        sys.exit(1)

    if not samples:
        logger.error("No samples loaded — nothing to evaluate")
        sys.exit(1)

    # ── Initialise worker ─────────────────────────────────────────────────
    if args.mock:
        with patch("src.workers.dermatology.DermatologyWorker._init_ollama"):
            worker = DermatologyWorker(use_ollama=True)
    else:
        logger.info("Initialising DermatologyWorker (requires Ollama)…")
        worker = DermatologyWorker(use_ollama=True)

    # ── Run evaluation ────────────────────────────────────────────────────
    logger.info("Starting evaluation on %d samples…", len(samples))
    metrics = run_evaluation(
        samples=samples,
        worker=worker,
        mock=args.mock,
        mock_correct_rate=args.mock_accuracy,
    )

    # ── Output ────────────────────────────────────────────────────────────
    print_report(metrics)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Results saved to %s", args.output)


if __name__ == "__main__":
    main()
