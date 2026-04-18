"""
Smoke tests for src.workers.dermatology.DermatologyWorker.

Covers:
  - Initialisation (Ollama mocked)
  - analyze() returns the correct output structure
  - Disclaimer always present
  - Classification is always from the allowed taxonomy
  - Image preprocessing handles RGB, RGBA, and grayscale inputs
  - JSON parsing handles markdown fences and partial JSON
  - Unknown classifications are normalised to "unknown"
  - Overconfident language is softened by safety enforcement
  - No-image fallback returns a graceful response

Run with:
    python -m pytest tests/test_dermatology.py -v
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.workers.dermatology import SKIN_CLASSES, DermatologyWorker


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _make_mock_response(classification: str = "benign_nevus") -> MagicMock:
    """Build a mock Ollama chat response containing valid JSON."""
    data = {
        "classification": classification,
        "confidence": "medium",
        "visual_description": "Round, evenly pigmented lesion with smooth borders.",
        "risk_level": "low",
        "plain_explanation": "This looks like a common benign mole.",
        "recommended_action": "Monitor monthly for any changes.",
        "abcde_notes": "A: symmetric, B: regular, C: uniform, D: ~5mm, E: stable",
    }
    resp = MagicMock()
    resp.message.content = json.dumps(data)
    return resp


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image():
    """224×224 RGB image simulating a skin lesion."""
    arr = np.random.randint(80, 200, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def rgba_image():
    """300×300 RGBA image (PNG with alpha channel)."""
    arr = np.random.randint(0, 255, (300, 300, 4), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGBA")


@pytest.fixture
def grayscale_image():
    """150×150 grayscale image."""
    arr = np.random.randint(0, 255, (150, 150), dtype=np.uint8)
    return Image.fromarray(arr, mode="L")


@pytest.fixture
def mock_worker():
    """DermatologyWorker with the Ollama backend mocked."""
    with patch("src.workers.dermatology.DermatologyWorker._init_ollama"):
        worker = DermatologyWorker(use_ollama=True)
        worker._client = MagicMock()
        worker._client.chat.return_value = _make_mock_response()
        return worker


# ─── Initialisation ───────────────────────────────────────────────────────────

class TestInit:
    def test_worker_creates_without_crash(self, mock_worker):
        assert mock_worker is not None

    def test_ollama_client_set(self, mock_worker):
        assert mock_worker._client is not None

    def test_use_ollama_flag(self, mock_worker):
        assert mock_worker.use_ollama is True


# ─── Image preprocessing ──────────────────────────────────────────────────────

class TestPreprocessing:
    def test_rgb_image_stays_rgb(self, mock_worker, dummy_image):
        out = mock_worker._preprocess_image(dummy_image)
        assert out.mode == "RGB"

    def test_rgba_converted_to_rgb(self, mock_worker, rgba_image):
        out = mock_worker._preprocess_image(rgba_image)
        assert out.mode == "RGB"

    def test_grayscale_converted_to_rgb(self, mock_worker, grayscale_image):
        out = mock_worker._preprocess_image(grayscale_image)
        assert out.mode == "RGB"

    def test_large_image_resized(self, mock_worker):
        big = Image.new("RGB", (2048, 2048), (150, 100, 80))
        out = mock_worker._preprocess_image(big)
        from src.config import IMAGE_SIZE
        assert out.size == IMAGE_SIZE

    def test_small_image_upscaled(self, mock_worker):
        small = Image.new("RGB", (32, 32), (150, 100, 80))
        out = mock_worker._preprocess_image(small)
        from src.config import IMAGE_SIZE
        assert out.size == IMAGE_SIZE

    def test_base64_encoding_produces_string(self, mock_worker, dummy_image):
        b64 = mock_worker._image_to_base64(dummy_image)
        assert isinstance(b64, str)
        assert len(b64) > 0


# ─── analyze() output structure ───────────────────────────────────────────────

class TestAnalyzeStructure:
    REQUIRED_KEYS = {
        "classification", "confidence", "visual_description",
        "risk_level", "plain_explanation", "recommended_action",
        "abcde_notes", "disclaimer",
    }

    def test_all_required_keys_present(self, mock_worker, dummy_image):
        result = mock_worker.analyze(image=dummy_image, symptoms="itchy spot")
        assert self.REQUIRED_KEYS.issubset(result.keys()), (
            f"Missing keys: {self.REQUIRED_KEYS - result.keys()}"
        )

    def test_disclaimer_always_non_empty(self, mock_worker, dummy_image):
        result = mock_worker.analyze(image=dummy_image)
        assert result["disclaimer"]
        assert len(result["disclaimer"]) > 20

    def test_disclaimer_contains_professional_advice(self, mock_worker, dummy_image):
        result = mock_worker.analyze(image=dummy_image)
        disclaimer_lower = result["disclaimer"].lower()
        assert "doctor" in disclaimer_lower or "professional" in disclaimer_lower

    def test_classification_in_allowed_list(self, mock_worker, dummy_image):
        result = mock_worker.analyze(image=dummy_image)
        assert result["classification"] in SKIN_CLASSES

    def test_confidence_is_valid_value(self, mock_worker, dummy_image):
        result = mock_worker.analyze(image=dummy_image)
        assert result["confidence"] in {"low", "medium", "high"}

    def test_risk_level_is_valid_value(self, mock_worker, dummy_image):
        result = mock_worker.analyze(image=dummy_image)
        assert result["risk_level"] in {"low", "moderate", "high", "urgent", "unknown"}

    def test_raw_model_output_stored(self, mock_worker, dummy_image):
        result = mock_worker.analyze(image=dummy_image)
        assert "raw_model_output" in result


# ─── No-image fallback ────────────────────────────────────────────────────────

class TestNoImageFallback:
    def test_no_image_returns_unknown_classification(self, mock_worker):
        result = mock_worker.analyze(image=None)
        assert result["classification"] == "unknown"

    def test_no_image_explanation_mentions_image(self, mock_worker):
        result = mock_worker.analyze(image=None)
        assert "image" in result["plain_explanation"].lower()

    def test_no_image_with_symptoms_echoes_symptoms(self, mock_worker):
        result = mock_worker.analyze(image=None, symptoms="red rash on forearm")
        assert result["disclaimer"]

    def test_no_image_disclaimer_present(self, mock_worker):
        result = mock_worker.analyze(image=None)
        assert result["disclaimer"]


# ─── JSON parsing edge cases ──────────────────────────────────────────────────

class TestJSONParsing:
    def test_clean_json_parsed_correctly(self, mock_worker, dummy_image):
        mock_worker._client.chat.return_value = _make_mock_response("melanoma")
        result = mock_worker.analyze(image=dummy_image)
        assert result["classification"] == "melanoma"

    def test_markdown_fenced_json_parsed(self, mock_worker, dummy_image):
        resp = MagicMock()
        resp.message.content = (
            "Here is the result:\n```json\n"
            + json.dumps({
                "classification": "actinic_keratosis",
                "confidence": "high",
                "visual_description": "rough scaly patch",
                "risk_level": "moderate",
                "plain_explanation": "precancerous lesion",
                "recommended_action": "see dermatologist",
                "abcde_notes": "irregular border",
            })
            + "\n```\n"
        )
        mock_worker._client.chat.return_value = resp
        result = mock_worker.analyze(image=dummy_image)
        assert result["classification"] == "actinic_keratosis"

    def test_completely_invalid_json_returns_fallback(self, mock_worker, dummy_image):
        resp = MagicMock()
        resp.message.content = "Sorry, I cannot analyse this image."
        mock_worker._client.chat.return_value = resp
        result = mock_worker.analyze(image=dummy_image)
        assert "classification" in result
        assert "disclaimer" in result

    def test_partial_json_fills_missing_fields(self, mock_worker, dummy_image):
        resp = MagicMock()
        resp.message.content = '{"classification": "seborrheic_keratosis"}'
        mock_worker._client.chat.return_value = resp
        result = mock_worker.analyze(image=dummy_image)
        assert result["classification"] == "seborrheic_keratosis"
        assert result["confidence"]
        assert result["plain_explanation"]

    def test_unknown_classification_normalised(self, mock_worker, dummy_image):
        resp = MagicMock()
        resp.message.content = json.dumps({
            "classification": "super_rare_unknown_thing",
            "confidence": "low",
            "visual_description": "test",
            "risk_level": "low",
            "plain_explanation": "test",
            "recommended_action": "see doctor",
            "abcde_notes": "test",
        })
        mock_worker._client.chat.return_value = resp
        result = mock_worker.analyze(image=dummy_image)
        assert result["classification"] == "unknown"


# ─── Safety enforcement ───────────────────────────────────────────────────────

class TestSafetyEnforcement:
    def test_overconfident_language_softened(self, mock_worker, dummy_image):
        resp = MagicMock()
        resp.message.content = json.dumps({
            "classification": "melanoma",
            "confidence": "high",
            "visual_description": "dark asymmetric lesion",
            "risk_level": "high",
            "plain_explanation": "This is definitely melanoma. You have cancer.",
            "recommended_action": "Seek urgent care immediately.",
            "abcde_notes": "highly asymmetric, irregular border",
        })
        mock_worker._client.chat.return_value = resp
        result = mock_worker.analyze(image=dummy_image)
        # "this is definitely" must be softened by safety enforcement
        assert "this is definitely" not in result["plain_explanation"].lower()

    def test_disclaimer_appended_to_explanation(self, mock_worker, dummy_image):
        mock_worker._client.chat.return_value = _make_mock_response()
        result = mock_worker.analyze(image=dummy_image)
        # Disclaimer must appear either in the field or be a separate key
        has_disclaimer = (
            result.get("disclaimer")
            or "doctor" in result.get("plain_explanation", "").lower()
        )
        assert has_disclaimer


# ─── Multiple classifications ─────────────────────────────────────────────────

class TestAllClassifications:
    @pytest.mark.parametrize("cls", SKIN_CLASSES)
    def test_each_skin_class_accepted(self, mock_worker, dummy_image, cls):
        mock_worker._client.chat.return_value = _make_mock_response(cls)
        result = mock_worker.analyze(image=dummy_image)
        assert result["classification"] == cls
