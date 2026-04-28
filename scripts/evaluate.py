"""
Evaluation script for G4G RuralClinic AI — DermatologyWorker.

Supports four data source modes:
  --dataset mock        Synthetic coloured images + injected responses.
                        No Ollama required. Use for CI and dry runs.
  --dataset isic        Real images in a class-named directory tree:
                            data/samples/melanoma/*.jpg
                            data/samples/benign_nevus/*.jpg  ...
  --dataset isic_csv    Flat image directory + ISIC ground-truth CSV
                            (one-hot or string label column).
  --dataset dermamnist  DermaMNIST (via the medmnist package).
                        Install: pip install medmnist
                        Downloaded automatically on first run.

Metrics produced per run:
  Accuracy              — fraction of exact-match classifications.
  Per-class P / R / F1  — standard binary-per-class metrics.
  Macro-average F1      — unweighted mean over classes with at least 1 sample.
  Safety adherence      — fraction of responses containing a medical disclaimer.
  Confusion matrix      — serialised as a JSON-safe dict of dicts.

Output:
  Printed summary table to stdout.
  Full metrics saved to --output (default: logs/eval_results.json).

Usage examples:

    # Dry run, no Ollama needed:
    python scripts/evaluate.py --dataset mock

    # Real ISIC samples (download first):
    bash scripts/download_sample_data.sh
    python scripts/evaluate.py --dataset isic --data-dir data/samples

    # DermaMNIST (auto-downloads ~40 MB):
    python scripts/evaluate.py --dataset dermamnist --max-samples 200

    # ISIC CSV format:
    python scripts/evaluate.py --dataset isic_csv \\
        --data-dir data/isic_images --csv data/isic/ISIC_2019_Training_GroundTruth.csv

Project: G4G RuralClinic AI — offline frontier AI for accessible
dermatology screening in rural and underserved communities.
"""

import argparse
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from PIL import Image

# Make the project root importable when run as a top-level script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.workers.dermatology import SKIN_CLASSES, DermatologyWorker
from src.config import LOGS_DIR, SAMPLE_DATA_DIR

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ─── DermaMNIST label mapping ─────────────────────────────────────────────────
# DermaMNIST uses the HAM10000 7-class taxonomy; we map to SKIN_CLASSES.
# Reference: https://medmnist.com/
DERMAMNIST_LABEL_MAP: dict[int, str] = {
    0: "actinic_keratosis",      # akiec
    1: "basal_cell_carcinoma",   # bcc
    2: "seborrheic_keratosis",   # bkl (benign keratosis-like lesions)
    3: "dermatofibroma",         # df
    4: "benign_nevus",           # nv (melanocytic nevi)
    5: "vascular_lesion",        # vasc
    6: "melanoma",               # mel
}

# ─── Mock response fixtures ───────────────────────────────────────────────────

def _mock_response(classification: str) -> dict:
    """
    Pre-canned correct-looking DermatologyWorker response for synthetic eval.
    Used only when --dataset mock is active.
    """
    low_risk_classes = {
        "benign_nevus", "normal_skin",
        "seborrheic_keratosis", "dermatofibroma",
    }
    return {
        "classification": classification,
        "confidence": "medium",
        "visual_description": f"Synthetic test image — class: {classification}.",
        "risk_level": "low" if classification in low_risk_classes else "moderate",
        "plain_explanation": (
            f"This is a synthetic test sample labelled {classification}. "
            f"In a real scenario, a qualified dermatologist should evaluate any skin concern."
        ),
        "recommended_action": "Consult a qualified dermatologist.",
        "abcde_notes": "Synthetic — not applicable to real clinical assessment.",
        "disclaimer": (
            "IMPORTANT: I am not a doctor. This AI-generated analysis is for "
            "educational and informational purposes only."
        ),
        "raw_model_output": "{}",
    }


# ─── Synthetic sample generation ─────────────────────────────────────────────

def build_synthetic_samples(n_per_class: int = 5) -> list[tuple[Image.Image, str]]:
    """
    Generate solid-colour 224×224 PIL images as stand-in test data.

    Each class gets a distinct colour to ensure the heuristic mock
    inject always produces a determinate result. Returns a list of
    (PIL Image, ground-truth label) tuples.
    """
    import numpy as np

    palette = [
        (180, 80, 80),   (80, 160, 80),  (80, 80, 180),
        (170, 170, 60),  (60, 170, 170), (170, 60, 170),
        (140, 110, 80),  (80, 140, 110), (110, 80, 140),
        (150, 150, 150),
    ]
    samples: list[tuple[Image.Image, str]] = []
    for cls, colour in zip(SKIN_CLASSES, palette):
        for _ in range(n_per_class):
            arr = np.full((224, 224, 3), colour, dtype=np.uint8)
            samples.append((Image.fromarray(arr), cls))
    return samples


# ─── Data loaders ─────────────────────────────────────────────────────────────

def load_directory_tree(
    data_dir: Path,
    max_per_class: Optional[int] = None,
) -> list[tuple[Path, str]]:
    """
    Load (image_path, label) pairs from:
        data_dir/
            melanoma/   img1.jpg  img2.jpg …
            benign_nevus/ …
    Only directories whose name is in SKIN_CLASSES are loaded.
    """
    samples: list[tuple[Path, str]] = []
    for cls in SKIN_CLASSES:
        cls_dir = data_dir / cls
        if not cls_dir.is_dir():
            continue
        imgs = (
            sorted(cls_dir.glob("*.jpg"))
            + sorted(cls_dir.glob("*.jpeg"))
            + sorted(cls_dir.glob("*.png"))
        )
        if max_per_class:
            imgs = imgs[:max_per_class]
        for p in imgs:
            samples.append((p, cls))
        if imgs:
            logger.info("  loaded %3d images from class: %s", len(imgs), cls)
    return samples


def load_isic_csv(
    data_dir: Path,
    csv_path: Path,
    max_samples: Optional[int] = None,
) -> list[tuple[Path, str]]:
    """
    Load (image_path, label) from an ISIC-style CSV with:
      - One-hot columns matching SKIN_CLASSES names, OR
      - A string 'label' or 'dx' column.

    Image files are resolved against data_dir with common extensions.
    """
    import csv

    samples: list[tuple[Path, str]] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        one_hot_cols = [f for f in fieldnames if f in SKIN_CLASSES]
        label_col = "label" if "label" in fieldnames else ("dx" if "dx" in fieldnames else None)

        for row in reader:
            img_name = row.get("image") or row.get("image_id") or ""
            if not img_name:
                continue

            img_path: Optional[Path] = None
            for ext in (".jpg", ".jpeg", ".png", ""):
                candidate = data_dir / (img_name + ext)
                if candidate.exists():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            if one_hot_cols:
                label = max(one_hot_cols, key=lambda c: float(row.get(c, 0) or 0))
            elif label_col:
                raw = row.get(label_col, "unknown").lower().replace(" ", "_")
                label = raw if raw in SKIN_CLASSES else "unknown"
            else:
                continue

            samples.append((img_path, label))
            if max_samples and len(samples) >= max_samples:
                break

    logger.info("Loaded %d samples from CSV %s", len(samples), csv_path)
    return samples


def load_dermamnist(
    max_samples: Optional[int] = None,
    split: str = "test",
) -> list[tuple[Image.Image, str]]:
    """
    Load DermaMNIST samples via the medmnist package.

    The package downloads the dataset (~40 MB) automatically on first run.
    Install: pip install medmnist

    Returns (PIL Image, label_str) tuples.
    """
    try:
        import medmnist
        from medmnist import DermaMNIST as _DermaMNIST
        import numpy as np
    except ImportError:
        logger.error(
            "medmnist not installed — run: pip install medmnist\n"
            "Alternatively use: --dataset mock (no install required)"
        )
        sys.exit(1)

    logger.info("Loading DermaMNIST split='%s' (downloads ~40 MB if not cached)…", split)
    dataset = _DermaMNIST(split=split, download=True, size=224)

    samples: list[tuple[Image.Image, str]] = []
    for i, (img_arr, label_arr) in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
        label_int = int(label_arr.item() if hasattr(label_arr, "item") else label_arr[0])
        label_str = DERMAMNIST_LABEL_MAP.get(label_int, "unknown")
        if isinstance(img_arr, Image.Image):
            pil_img = img_arr.convert("RGB")
        else:
            pil_img = Image.fromarray(
                (img_arr.numpy() * 255).astype("uint8")
                if hasattr(img_arr, "numpy") else img_arr
            ).convert("RGB")
        samples.append((pil_img, label_str))

    logger.info("DermaMNIST: loaded %d samples", len(samples))
    return samples


# ─── Metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(
    predictions: list[str],
    ground_truth: list[str],
    classes: list[str],
    raw_results: Optional[list[dict]] = None,
) -> dict:
    """
    Compute classification and safety metrics.

    Args:
        predictions:  List of predicted classification strings.
        ground_truth: List of ground-truth label strings (same length).
        classes:      Full class vocabulary (SKIN_CLASSES).
        raw_results:  Optional list of raw DermatologyWorker result dicts;
                      used to compute safety_adherence_rate.

    Returns:
        dict suitable for JSON serialisation.
    """
    n = len(predictions)
    if n == 0:
        return {"error": "No predictions to evaluate"}

    accuracy = sum(p == g for p, g in zip(predictions, ground_truth)) / n

    per_class: dict[str, dict] = {}
    f1_scores: list[float] = []
    for cls in classes:
        tp = sum(p == cls and g == cls for p, g in zip(predictions, ground_truth))
        fp = sum(p == cls and g != cls for p, g in zip(predictions, ground_truth))
        fn = sum(p != cls and g == cls for p, g in zip(predictions, ground_truth))
        support = tp + fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )
        per_class[cls] = {
            "precision": round(precision, 4),
            "recall":    round(recall, 4),
            "f1":        round(f1, 4),
            "support":   support,
        }
        if support > 0:
            f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for pred, gt in zip(predictions, ground_truth):
        confusion[gt][pred] += 1
    confusion_dict = {k: dict(v) for k, v in confusion.items()}

    # Safety adherence: does every response contain a medical disclaimer?
    safety_rate: Optional[float] = None
    if raw_results:
        disclaimer_marker = "not a doctor"
        adherent = sum(
            1 for r in raw_results
            if disclaimer_marker in (r.get("disclaimer", "") + r.get("plain_explanation", "")).lower()
        )
        safety_rate = adherent / len(raw_results)

    result = {
        "n_samples":   n,
        "accuracy":    round(accuracy, 4),
        "macro_f1":    round(macro_f1, 4),
        "per_class":   per_class,
        "confusion_matrix": confusion_dict,
    }
    if safety_rate is not None:
        result["safety_adherence_rate"] = round(safety_rate, 4)
    return result


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_evaluation(
    samples,
    worker: DermatologyWorker,
    mock: bool = False,
    mock_correct_rate: float = 0.70,
) -> dict:
    """
    Run inference on all samples and collect predictions + raw results.

    Args:
        samples:           list of (image_or_path, label) pairs.
        worker:            DermatologyWorker instance.
        mock:              If True, inject pre-canned responses without LLM.
        mock_correct_rate: Fraction of injected responses that match ground truth.

    Returns:
        Metrics dict from compute_metrics().
    """
    random.seed(42)

    predictions: list[str] = []
    ground_truth: list[str] = []
    raw_results:  list[dict] = []
    errors = 0

    for i, (img_or_path, label) in enumerate(samples):
        # Resolve image
        if isinstance(img_or_path, Path):
            try:
                image = Image.open(img_or_path).convert("RGB")
            except Exception as exc:
                logger.warning("Cannot open %s: %s", img_or_path, exc)
                errors += 1
                continue
        else:
            image = img_or_path

        # Inference
        if mock:
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

        predictions.append(result.get("classification", "unknown"))
        ground_truth.append(label)
        raw_results.append(result)

        if (i + 1) % 20 == 0:
            logger.info("  Progress: %d/%d", i + 1, len(samples))

    if errors:
        logger.warning("%d samples skipped due to errors", errors)

    metrics = compute_metrics(predictions, ground_truth, SKIN_CLASSES, raw_results)
    metrics["errors_skipped"] = errors
    return metrics


# ─── Report printing ──────────────────────────────────────────────────────────

def print_report(metrics: dict) -> None:
    print("\n" + "═" * 62)
    print("  G4G RuralClinic AI — Evaluation Report")
    print("═" * 62)
    print(f"  Samples evaluated    : {metrics.get('n_samples', 0)}")
    print(f"  Accuracy             : {metrics.get('accuracy', 0):.1%}")
    print(f"  Macro F1             : {metrics.get('macro_f1', 0):.1%}")
    if "safety_adherence_rate" in metrics:
        print(f"  Safety adherence     : {metrics['safety_adherence_rate']:.1%}")
    if metrics.get("errors_skipped"):
        print(f"  Errors skipped       : {metrics['errors_skipped']}")
    print()
    print(f"  {'Class':<32} {'P':>6} {'R':>6} {'F1':>6} {'N':>5}")
    print("  " + "─" * 57)
    for cls, m in (metrics.get("per_class") or {}).items():
        if m["support"] > 0:
            print(
                f"  {cls:<32} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                f"{m['f1']:>6.3f} {m['support']:>5}"
            )
    print("═" * 62 + "\n")


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate G4G RuralClinic AI DermatologyWorker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["mock", "isic", "isic_csv", "dermamnist"],
        default="mock",
        help="Data source.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=SAMPLE_DATA_DIR,
        help="Root directory for isic/isic_csv datasets.",
    )
    parser.add_argument(
        "--csv", type=Path, default=None,
        help="Ground-truth CSV path for isic_csv mode.",
    )
    parser.add_argument(
        "--dermamnist-split",
        choices=["train", "val", "test"],
        default="test",
        help="DermaMNIST dataset split.",
    )
    parser.add_argument(
        "--max-samples", type=int, default=200,
        help="Maximum total samples to evaluate.",
    )
    parser.add_argument(
        "--max-per-class", type=int, default=None,
        help="Maximum samples per class (isic / isic_csv only).",
    )
    parser.add_argument(
        "--mock-accuracy", type=float, default=0.70,
        help="Fraction of correct mock predictions (mock mode only).",
    )
    parser.add_argument(
        "--output", type=Path, default=LOGS_DIR / "eval_results.json",
        help="JSON output file path.",
    )
    args = parser.parse_args()

    # ── Load samples ──────────────────────────────────────────────────────
    if args.dataset == "mock":
        logger.info("Mock mode: generating synthetic samples")
        n_per = max(1, args.max_samples // len(SKIN_CLASSES))
        samples = build_synthetic_samples(n_per_class=n_per)
        logger.info("Generated %d synthetic samples across %d classes", len(samples), len(SKIN_CLASSES))
        is_mock = True

    elif args.dataset == "dermamnist":
        samples = load_dermamnist(
            max_samples=args.max_samples,
            split=args.dermamnist_split,
        )
        is_mock = False

    elif args.dataset == "isic_csv":
        if not (args.csv and args.csv.exists()):
            logger.error("--csv is required and must exist for isic_csv mode")
            sys.exit(1)
        samples = load_isic_csv(
            args.data_dir,
            args.csv,
            max_samples=args.max_samples,
        )
        is_mock = False

    else:  # isic
        if not args.data_dir.is_dir():
            logger.error(
                "Directory not found: %s\n"
                "Download sample data with: bash scripts/download_sample_data.sh\n"
                "Or use --dataset mock for a no-data dry run.",
                args.data_dir,
            )
            sys.exit(1)
        max_per = args.max_per_class or (args.max_samples // len(SKIN_CLASSES))
        samples = load_directory_tree(args.data_dir, max_per_class=max_per)
        if args.max_samples:
            samples = samples[: args.max_samples]
        logger.info("Loaded %d samples", len(samples))
        is_mock = False

    if not samples:
        logger.error("No samples loaded — nothing to evaluate")
        sys.exit(1)

    # ── Initialise worker ─────────────────────────────────────────────────
    if is_mock:
        with patch("src.workers.dermatology.DermatologyWorker._init_ollama"):
            worker = DermatologyWorker(use_ollama=True)
        logger.info("Worker initialised in mock mode (no Ollama)")
    else:
        logger.info("Initialising DermatologyWorker (requires running Ollama)…")
        worker = DermatologyWorker(use_ollama=True)

    # ── Run ───────────────────────────────────────────────────────────────
    logger.info("Starting evaluation on %d samples…", len(samples))
    metrics = run_evaluation(
        samples=samples,
        worker=worker,
        mock=is_mock,
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
