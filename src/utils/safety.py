"""
Safety utilities for G4G RuralClinic AI.

All AI-generated text must pass through these helpers before being shown
to users. Core rules:
  1. Soften overconfident diagnostic language.
  2. Always append the medical disclaimer.
  3. Validate that required safety fields are present in structured outputs.
  4. Enhance compassion and accessibility for low-health-literacy users.
  5. Add clear referral timing guidance.

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
    (r"\bis cancer\b",                  "might be a serious condition that needs checking"),
    (r"\bmelanoma\b",                   "a type of skin cancer (melanoma)"),
    (r"\bcarcinoma\b",                  "a type of skin cancer (carcinoma)"),
    (r"\byou need surgery\b",           "surgery might be an option to discuss with your doctor"),
    (r"\byou should\s+(not )?see a doctor\b", r"it is important to speak with a healthcare professional"),
    (r"\bdefinitely (?:cancer|melanoma|carcinoma)\b", 
     "possible serious condition that needs professional evaluation"),
    (r"\bthis (?:is|looks like) (?:a )?tumor\b", "this shows a growth that needs evaluation"),
    (r"\bit's (?:cancer|melanoma)\b",   "it could be a serious condition"),
    (r"\bthe (?:lesion|mole) (?:is )?(?:cancer|melanoma)\b", 
     "the growth may indicate a serious condition"),
    (r"\byou must\b",                    "consider discussing with your clinician"),
    (r"\byou require\b",                 "you may want to discuss with your clinician"),
    (r"\bemergency\b",                   "situation needing prompt attention"),
]

# Patterns for adding compassionate / low-literacy softening
_COMPASSIONATE_REPLACEMENTS = [
    (r"\bthe worst\b",                   "a very serious"),
    (r"\bhorrible\b",                    "concerning"),
    (r"\bterrible\b",                    "concerning"),
    (r"\bdangerous\b",                   "needing prompt attention"),
    (r"\bscary\b",                       "worrying"),
    (r"\balarming\b",                    "concerning"),
    (r"\bterrible and scary\b",          "concerning"),
    (r"\bconcerning and worrying\b",     "concerning"),
]

# Referral timing guidance — soften urgency but maintain clarity
_REFERRAL_REPLACEMENTS = [
    (r"\bgo to the (?:emergency|ER)\b",  "seek medical care right away"),
    (r"\bER\b",                           "emergency department"),
    (r"\bsee a doctor today\b",          "speak with a healthcare professional soon"),
    (r"\bimmediately\b",                 "right away"),
    (r"\bas soon as possible\b",         "when you can"),
    (r"\bright away right away\b",       "right away"),
]

# Combined replacements applied in sequence
_ALL_REPLACEMENTS = _OVERCONFIDENT_REPLACEMENTS + _COMPASSIONATE_REPLACEMENTS + _REFERRAL_REPLACEMENTS

# Post-processing: clean up duplicate/repeated words that may result from multiple replacements
_CLEANUP_REPLACEMENTS = [
    (r"\b(right away)( right away)\b", r"\1"),
    (r"\b(seek medical care right away today)\b", "seek medical care right away"),
]


def enforce_safety(text: str) -> str:
    """
    Apply all safety rules to a text string:
      1. Replace overconfident diagnostic language with hedged alternatives.
      2. Add compassionate wording for low-health-literacy users.
      3. Soften referral timing while maintaining clarity.
      4. Append the standard disclaimer if not already present.

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

    text = _soften_language(text)

    if _DISCLAIMER_MARKER not in text:
        text = f"{text}\n\n{SAFETY_DISCLAIMER}"

    return text


def _soften_language(text: str) -> str:
    """Replace overconfident diagnostic phrases with appropriately hedged ones."""
    # Apply all three categories: overconfident, compassionate, referral
    for pattern, replacement in _ALL_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Apply cleanup fixes
    for pattern, replacement in _CLEANUP_REPLACEMENTS:
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
    Uses plain, compassionate language accessible to users with low health literacy.

    Args:
        result: Validated dict from DermatologyWorker.analyze().

    Returns:
        Markdown-formatted string for UI display.
    """
    lines = []

    classification = result.get("classification", "unknown")
    confidence = result.get("confidence", "unknown")
    risk = result.get("risk_level", "unknown")

    # Format labels clearly and accessibly
    lines.append(f"**What was found:** {classification.replace('_', ' ').title()}")
    lines.append(f"**Confidence level:** {confidence.title()}")
    lines.append(f"**Risk level:** {risk.title()}")
    lines.append("")

    if result.get("visual_description"):
        lines.append(f"**What it looks like:**")
        lines.append(result["visual_description"])
        lines.append("")

    if result.get("abcde_notes"):
        lines.append(f"**ABCDE assessment details:**")
        lines.append(result["abcde_notes"])
        lines.append("")

    if result.get("plain_explanation"):
        lines.append(f"**What this means for you:**")
        lines.append(result["plain_explanation"])
        lines.append("")

    if result.get("recommended_action"):
        lines.append(f"**Suggested next step:**")
        lines.append(result["recommended_action"])
        lines.append("")

    lines.append("---")
    lines.append(result.get("disclaimer", SAFETY_DISCLAIMER))

    return "\n".join(lines)
