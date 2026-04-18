"""
Safety utilities for G4G RuralClinic AI.

All AI-generated text must pass through these helpers before being shown
to users. Core rules:
  1. Soften overconfident diagnostic language.
  2. Always append the medical disclaimer.
  3. Validate that required safety fields are present in structured outputs.

Project: G4G RuralClinic AI — offline dermatology assistant.
These guardrails are non-negotiable; removing them would make the system
unsafe for its intended use in rural/underserved healthcare contexts.
"""

import re
import logging
from src.config import SAFETY_DISCLAIMER

logger = logging.getLogger(__name__)

# Marker used to detect if disclaimer is already embedded in a response.
_DISCLAIMER_MARKER = "I am not a doctor"

# Patterns that imply definitive diagnosis — we replace with hedged alternatives.
# Order matters: more specific patterns first.
_OVERCONFIDENT_REPLACEMENTS = [
    (r"\bthis is definitely\b",         "this appears to be"),
    (r"\bthis is certainly\b",          "this may be consistent with"),
    (r"\bthis is clearly\b",            "this could be"),
    (r"\bI(?:'m| am) certain\b",        "based on the image, it is possible"),
    (r"\bI diagnose\b",                 "the analysis suggests"),
    (r"\bdiagnosed as\b",               "appears consistent with"),
    (r"\bconfirmed (?:as|to be)\b",     "appears to be"),
    (r"\byou have (?:cancer|melanoma)\b", "this warrants urgent professional evaluation"),
    (r"\b100%\s+(?:sure|certain)\b",    "with some confidence"),
    (r"\bno doubt\b",                   "possibly"),
    (r"\bwithout question\b",           "likely"),
]


def enforce_safety(text: str) -> str:
    """
    Apply all safety rules to a text string:
      1. Replace overconfident diagnostic language with hedged alternatives.
      2. Append the standard disclaimer if not already present.

    Args:
        text: Raw model or worker output string.

    Returns:
        Safety-enforced string, always ending with the disclaimer.
    """
    if not text or not text.strip():
        return text

    text = _soften_language(text)

    if _DISCLAIMER_MARKER not in text:
        text = f"{text}\n\n{SAFETY_DISCLAIMER}"

    return text


def _soften_language(text: str) -> str:
    """Replace overconfident diagnostic phrases with appropriately hedged ones."""
    for pattern, replacement in _OVERCONFIDENT_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def validate_output_structure(result: dict) -> bool:
    """
    Check that a worker result contains the minimum required safety fields.

    Args:
        result: Dict returned by a specialist worker.

    Returns:
        True if all required safety fields are present and non-empty.
    """
    required_keys = ["classification", "confidence", "plain_explanation", "disclaimer"]
    missing = [k for k in required_keys if not result.get(k)]
    if missing:
        logger.warning("Worker output missing safety fields: %s", missing)
        return False
    return True


def format_for_display(result: dict) -> str:
    """
    Convert a structured worker result dict into a clean, user-facing
    markdown string suitable for display in the Gradio UI.

    Args:
        result: Validated dict from DermatologyWorker.analyze().

    Returns:
        Markdown-formatted string for UI display.
    """
    lines = []

    classification = result.get("classification", "unknown")
    confidence = result.get("confidence", "unknown")
    risk = result.get("risk_level", "unknown")

    lines.append(f"**Classification:** {classification.replace('_', ' ').title()}")
    lines.append(f"**Confidence:** {confidence.title()}")
    lines.append(f"**Risk Level:** {risk.title()}")
    lines.append("")

    if result.get("visual_description"):
        lines.append(f"**Visual Observations:**")
        lines.append(result["visual_description"])
        lines.append("")

    if result.get("abcde_notes"):
        lines.append(f"**ABCDE Assessment:**")
        lines.append(result["abcde_notes"])
        lines.append("")

    if result.get("plain_explanation"):
        lines.append(f"**What This Means:**")
        lines.append(result["plain_explanation"])
        lines.append("")

    if result.get("recommended_action"):
        lines.append(f"**Recommended Action:**")
        lines.append(result["recommended_action"])
        lines.append("")

    lines.append("---")
    lines.append(result.get("disclaimer", SAFETY_DISCLAIMER))

    return "\n".join(lines)
