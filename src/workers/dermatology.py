"""
Dermatology specialist worker for G4G RuralClinic AI.

Uses Gemma 4's native multimodal (image + text) capabilities to analyze
skin lesion images and return a structured informational assessment.

Design principles:
  - Safety first: every output includes a strong medical disclaimer.
  - Never diagnose: always hedge with "appears consistent with", "may be".
  - Offline compatible: works via Ollama (preferred) or HF pipeline.
  - Structured output: returns a consistent dict for orchestrator synthesis.
  - No ML training required: few-shot examples baked into the prompt.

Project: G4G RuralClinic AI — offline frontier AI for accessible
dermatology screening in rural and underserved communities.
"""

import base64
import io
import json
import logging
import re
from typing import Optional

from PIL import Image

from src.config import (
    HF_MODEL_NAME,
    IMAGE_SIZE,
    MODEL_NAME,
    OFFLINE_MODE,
    OLLAMA_HOST,
    SAFETY_DISCLAIMER,
)
from src.utils.safety import enforce_safety

logger = logging.getLogger(__name__)


# ─── Classification taxonomy ──────────────────────────────────────────────────
# Based on the HAM10000 / ISIC taxonomy used in standard derm datasets.
SKIN_CLASSES = [
    "melanoma",
    "basal_cell_carcinoma",
    "squamous_cell_carcinoma",
    "benign_nevus",
    "seborrheic_keratosis",
    "actinic_keratosis",
    "dermatofibroma",
    "vascular_lesion",
    "normal_skin",
    "unknown",
]

# ─── Few-shot examples (no training, knowledge-grounded prompt) ───────────────
_FEW_SHOT_CONTEXT = """
REFERENCE EXAMPLES (use as calibration only — do not copy):

Example A — Melanoma suspicion:
  Image: Asymmetric dark lesion, irregular notched border, multiple shades
         of brown/black/red, diameter appears >6mm.
  → classification: melanoma, risk: high, confidence: high
  → Action: Urgent dermatologist referral.

Example B — Benign nevus:
  Image: Round, uniformly brown mole, smooth regular border, single color,
         approximately 4mm, stable appearance.
  → classification: benign_nevus, risk: low, confidence: medium
  → Action: Monitor monthly; see doctor if it changes.

Example C — Actinic keratosis:
  Image: Rough, scaly patch on sun-exposed area, slightly raised, pink/red,
         irregular but relatively small.
  → classification: actinic_keratosis, risk: moderate, confidence: medium
  → Action: Dermatologist evaluation within 4 weeks (precancerous).

Example D — Seborrheic keratosis:
  Image: Stuck-on-looking waxy brown plaque, well-defined borders, uniform
         texture, rough surface.
  → classification: seborrheic_keratosis, risk: low, confidence: medium
  → Action: Benign; monitor for rapid growth or irritation.
"""

# ─── Analysis prompt template ─────────────────────────────────────────────────
_ANALYSIS_PROMPT = """You are assisting a dermatology screening tool for rural, underserved communities.
Analyze the provided skin lesion image carefully and objectively.

{few_shot}

Patient-reported symptoms / context: {symptoms}

Provide a structured analysis as a single valid JSON object with EXACTLY these keys:
{{
  "classification": "<one of: {classes}>",
  "confidence": "<low | medium | high>",
  "visual_description": "<objective description: shape, color, borders, texture, size estimate>",
  "risk_level": "<low | moderate | high | urgent>",
  "plain_explanation": "<2–3 sentences in plain language for a non-medical person>",
  "recommended_action": "<specific next step, e.g. 'monitor monthly', 'see dermatologist within 2 weeks', 'seek urgent care today'>",
  "abcde_notes": "<ABCDE rule: Asymmetry / Border / Color / Diameter / Evolution observations>"
}}

RULES:
- Output ONLY the JSON object. No preamble, no trailing text, no markdown fences.
- Never claim certainty. Use hedging: "appears consistent with", "may suggest", "could indicate".
- If the image quality is poor or inconclusive, set confidence to "low" and say so in plain_explanation.
- classification must be exactly one value from the allowed list.
"""


class DermatologyWorker:
    """
    Specialist worker for skin lesion analysis using Gemma 4 multimodal.

    Supports two inference backends:
      - Ollama (preferred): fast, CPU-optimised, runs locally after `ollama pull`
      - HuggingFace pipeline (fallback): uses transformers image-text-to-text

    The worker preprocesses the image, builds a structured prompt with few-shot
    examples, runs inference, and parses the JSON output into a validated dict.
    """

    def __init__(self, use_ollama: bool = True):
        """
        Initialise the dermatology worker.

        Args:
            use_ollama: Use Ollama backend if True, HF pipeline if False.
        """
        self.use_ollama = use_ollama
        self._client = None
        self._model = None
        self._processor = None

        if use_ollama:
            self._init_ollama()
        else:
            self._init_hf_pipeline()

    # ── Backend initialisation ────────────────────────────────────────────────

    def _init_ollama(self):
        """Connect to the local Ollama instance."""
        try:
            import ollama
            self._client = ollama.Client(host=OLLAMA_HOST)
            logger.info("DermatologyWorker: Ollama client ready at %s", OLLAMA_HOST)
        except Exception as exc:
            logger.error("DermatologyWorker: Ollama init failed (%s) — will use HF", exc)
            self.use_ollama = False

    def _init_hf_pipeline(self):
        """Load Gemma 4 multimodal pipeline via HuggingFace transformers."""
        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            logger.info("DermatologyWorker: loading HF model %s on %s…", HF_MODEL_NAME, device)
            self._processor = AutoProcessor.from_pretrained(
                HF_MODEL_NAME, local_files_only=OFFLINE_MODE
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                HF_MODEL_NAME,
                torch_dtype=dtype,
                device_map=device,
                local_files_only=OFFLINE_MODE,
            )
            logger.info("DermatologyWorker: HF model loaded on %s", device)
        except Exception as exc:
            logger.error("DermatologyWorker: HF pipeline init failed: %s", exc)
            raise RuntimeError(
                f"Cannot load Gemma 4. Ensure Ollama is running or the HF model is "
                f"downloaded. Original error: {exc}"
            ) from exc

    # ── Main analysis method ──────────────────────────────────────────────────

    def analyze(self, image: Optional[Image.Image], symptoms: str = "") -> dict:
        """
        Analyse a skin lesion image and return a structured assessment dict.

        Args:
            image: PIL Image of the skin lesion. May be None (text-only fallback).
            symptoms: Patient-reported symptom description (free text).

        Returns:
            dict with keys:
              classification, confidence, visual_description, risk_level,
              plain_explanation, recommended_action, abcde_notes,
              disclaimer, raw_model_output (debug).
        """
        logger.info(
            "DermatologyWorker.analyze | has_image=%s | symptoms=%d chars",
            image is not None,
            len(symptoms),
        )

        if image is None:
            return self._no_image_response(symptoms)

        processed = self._preprocess_image(image)

        prompt = _ANALYSIS_PROMPT.format(
            few_shot=_FEW_SHOT_CONTEXT,
            symptoms=symptoms.strip() if symptoms else "None provided.",
            classes=", ".join(SKIN_CLASSES),
        )

        if self.use_ollama and self._client:
            raw = self._run_ollama(processed, prompt)
        elif self._model is not None:
            raw = self._run_hf(processed, prompt)
        else:
            logger.error("No inference backend available.")
            return self._error_response("No inference backend available.")

        result = self._parse_output(raw)
        result["raw_model_output"] = raw
        result["disclaimer"] = SAFETY_DISCLAIMER

        # Enforce safety on all free-text fields
        result["plain_explanation"] = enforce_safety(result.get("plain_explanation", ""))
        result["recommended_action"] = enforce_safety(result.get("recommended_action", ""))

        logger.info(
            "DermatologyWorker result | cls=%s | conf=%s | risk=%s",
            result.get("classification"),
            result.get("confidence"),
            result.get("risk_level"),
        )
        return result

    # ── Image preprocessing ───────────────────────────────────────────────────

    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Normalise image to RGB and resize to IMAGE_SIZE.
        Handles RGBA, grayscale, and palette-mode inputs.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        logger.debug("Image preprocessed to %s %s", image.size, image.mode)
        return image

    def _image_to_base64(self, image: Image.Image) -> str:
        """Encode a PIL Image as a JPEG base64 string for the Ollama API."""
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    # ── Inference backends ────────────────────────────────────────────────────

    def _run_ollama(self, image: Image.Image, prompt: str) -> str:
        """Run multimodal inference via the local Ollama server."""
        image_b64 = self._image_to_base64(image)
        response = self._client.chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            options={
                "temperature": 0.1,   # Low temp → consistent structured JSON
                "num_predict": 600,
            },
        )
        return response.message.content

    def _run_hf(self, image: Image.Image, prompt: str) -> str:
        """Run multimodal inference via HuggingFace transformers pipeline."""
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_input = self._processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = self._processor(text=text_input, images=[image], return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=600,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self._processor.decode(new_tokens, skip_special_tokens=True)

    # ── Output parsing ────────────────────────────────────────────────────────

    def _parse_output(self, raw: str) -> dict:
        """
        Extract and validate the JSON object from raw model output.
        Handles markdown fences, leading/trailing text, and partial JSON.
        """
        # Strip optional ```json ... ``` fences
        raw_clean = re.sub(r"```(?:json)?", "", raw).strip()

        match = re.search(r"\{[\s\S]*\}", raw_clean)
        if match:
            try:
                parsed = json.loads(match.group())
                return self._fill_defaults(parsed)
            except json.JSONDecodeError as exc:
                logger.warning("JSON decode error: %s", exc)

        logger.warning("Could not parse JSON from model output; using fallback.")
        return self._fallback_result(raw)

    def _fill_defaults(self, parsed: dict) -> dict:
        """Ensure all required keys exist and classification is from the allowed list."""
        defaults = {
            "classification": "unknown",
            "confidence": "low",
            "visual_description": "Unable to determine from the provided image.",
            "risk_level": "unknown",
            "plain_explanation": (
                "The analysis could not be completed. Please consult a dermatologist."
            ),
            "recommended_action": "Consult a qualified dermatologist.",
            "abcde_notes": "Manual clinical evaluation required.",
        }
        for key, default in defaults.items():
            if not parsed.get(key):
                parsed[key] = default

        cls = parsed.get("classification", "").lower().replace(" ", "_")
        parsed["classification"] = cls if cls in SKIN_CLASSES else "unknown"

        return parsed

    def _fallback_result(self, raw: str) -> dict:
        """Return a safe, fully-populated result when output cannot be parsed."""
        return {
            "classification": "unknown",
            "confidence": "low",
            "visual_description": "Analysis could not be structured.",
            "risk_level": "unknown",
            "plain_explanation": (
                "The AI was unable to produce a structured analysis from this image. "
                "Please ensure the image is clear and well-lit, then try again. "
                "Consult a dermatologist for a proper evaluation."
            ),
            "recommended_action": "Consult a qualified dermatologist.",
            "abcde_notes": "Clinical evaluation required.",
            "parse_error": True,
            "raw_text": raw[:500],
        }

    def _no_image_response(self, symptoms: str) -> dict:
        """Response when analyze() is called without an image."""
        return {
            "classification": "unknown",
            "confidence": "low",
            "visual_description": "No image provided.",
            "risk_level": "unknown",
            "plain_explanation": (
                "No image was provided. To get a visual assessment, please upload "
                "a clear, well-lit photo of the skin area. "
                + (f"Described symptoms: {symptoms[:120]}." if symptoms else "")
            ),
            "recommended_action": (
                "Upload a clear image of the skin area and consult a dermatologist."
            ),
            "abcde_notes": "Cannot assess without an image.",
            "disclaimer": SAFETY_DISCLAIMER,
        }

    def _error_response(self, message: str) -> dict:
        """Return a safe error result when inference completely fails."""
        return {
            "classification": "unknown",
            "confidence": "low",
            "visual_description": "Error during analysis.",
            "risk_level": "unknown",
            "plain_explanation": f"Analysis failed: {message} Please consult a dermatologist.",
            "recommended_action": "Consult a qualified dermatologist.",
            "abcde_notes": "Error — clinical evaluation required.",
            "disclaimer": SAFETY_DISCLAIMER,
            "error": message,
        }
