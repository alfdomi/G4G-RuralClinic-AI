"""
Smoke tests for src.orchestrator.core.Orchestrator.

Covers:
  - Initialisation (Ollama mocked out)
  - route_and_run() returns the correct output structure
  - Disclaimer is always present in every response
  - Function-call log is populated when a worker is invoked
  - Image presence always triggers the dermatology worker
  - Conversation history is updated and can be cleared
  - Keyword-based routing fallback works correctly
  - Error paths return safe responses

Run with:
    python -m pytest tests/test_orchestrator.py -v
"""

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dummy_image():
    """224×224 RGB test image (random noise)."""
    arr = np.random.randint(80, 200, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture
def mock_orchestrator():
    """
    Orchestrator with the Ollama backend mocked so no local server is needed.
    The chat() mock returns a valid JSON dermatology response.
    """
    with patch("src.orchestrator.core.Orchestrator._init_ollama"):
        with patch("src.workers.dermatology.DermatologyWorker._init_ollama"):
            from src.orchestrator.core import Orchestrator

            orch = Orchestrator(use_ollama=True)
            orch._client = MagicMock()

            # Default: model returns no tool call → general response
            general_resp = MagicMock()
            general_resp.message.content = (
                "That is a good question about skin health. "
                "Please consult a dermatologist for proper evaluation."
            )
            general_resp.message.tool_calls = None
            orch._client.chat.return_value = general_resp

            return orch


def _worker_result_fixture():
    """Standard valid worker result dict."""
    return {
        "classification": "benign_nevus",
        "confidence": "medium",
        "visual_description": "Round, evenly pigmented brown lesion with regular borders.",
        "risk_level": "low",
        "plain_explanation": "This appears to be a common benign mole. Not immediately alarming.",
        "recommended_action": "Monitor monthly for any changes in size, colour, or shape.",
        "abcde_notes": "A: symmetric, B: regular, C: uniform brown, D: ~4mm est, E: stable",
        "disclaimer": "I am not a doctor. Consult a professional.",
        "raw_model_output": "{}",
    }


# ─── Initialisation tests ──────────────────────────────────────────────────────

class TestOrchestratorInit:
    def test_dermatology_worker_created(self, mock_orchestrator):
        from src.workers.dermatology import DermatologyWorker
        assert isinstance(mock_orchestrator.dermatology_worker, DermatologyWorker)

    def test_history_empty_on_start(self, mock_orchestrator):
        assert mock_orchestrator.history == []

    def test_function_call_log_empty_on_start(self, mock_orchestrator):
        assert mock_orchestrator.function_call_log == []


# ─── route_and_run output structure ───────────────────────────────────────────

class TestRouteAndRunStructure:
    REQUIRED_KEYS = {"response", "worker_result", "function_calls", "disclaimer", "timestamp"}

    def test_returns_all_required_keys(self, mock_orchestrator):
        result = mock_orchestrator.route_and_run("What is melanoma?")
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_disclaimer_never_empty(self, mock_orchestrator, dummy_image):
        result = mock_orchestrator.route_and_run("Check this spot", image=dummy_image)
        assert result["disclaimer"]
        assert len(result["disclaimer"]) > 20

    def test_response_is_non_empty_string(self, mock_orchestrator):
        result = mock_orchestrator.route_and_run("Tell me about eczema")
        assert isinstance(result["response"], str)
        assert len(result["response"]) > 0

    def test_function_calls_is_list(self, mock_orchestrator, dummy_image):
        with patch.object(
            mock_orchestrator.dermatology_worker, "analyze",
            return_value=_worker_result_fixture()
        ):
            result = mock_orchestrator.route_and_run("skin image", image=dummy_image)
        assert isinstance(result["function_calls"], list)

    def test_timestamp_present_and_non_empty(self, mock_orchestrator):
        result = mock_orchestrator.route_and_run("hello")
        assert result["timestamp"]


# ─── Routing behaviour ─────────────────────────────────────────────────────────

class TestRouting:
    def test_image_always_routes_to_derm_worker(self, mock_orchestrator, dummy_image):
        with patch.object(
            mock_orchestrator.dermatology_worker, "analyze",
            return_value=_worker_result_fixture()
        ) as mock_analyze:
            mock_orchestrator.route_and_run("Analyse this", image=dummy_image)
        mock_analyze.assert_called_once()

    def test_image_sets_worker_result(self, mock_orchestrator, dummy_image):
        with patch.object(
            mock_orchestrator.dermatology_worker, "analyze",
            return_value=_worker_result_fixture()
        ):
            result = mock_orchestrator.route_and_run("Analyse this", image=dummy_image)
        assert result["worker_result"] is not None
        assert result["worker_result"]["classification"] == "benign_nevus"

    def test_no_image_no_keyword_no_worker(self, mock_orchestrator):
        result = mock_orchestrator.route_and_run("What is the weather today?")
        assert result["worker_result"] is None

    def test_function_call_logged_when_worker_runs(self, mock_orchestrator, dummy_image):
        with patch.object(
            mock_orchestrator.dermatology_worker, "analyze",
            return_value=_worker_result_fixture()
        ):
            result = mock_orchestrator.route_and_run("check mole", image=dummy_image)
        assert len(result["function_calls"]) == 1
        assert result["function_calls"][0]["tool"] == "analyze_skin_lesion"


# ─── Keyword routing ───────────────────────────────────────────────────────────

class TestKeywordRouting:
    @pytest.mark.parametrize("query", [
        "I have a weird skin spot",
        "my mole has changed colour",
        "there is a rash on my arm",
        "dark lesion appeared last week",
        "what is this growth on my back",
    ])
    def test_derm_keywords_route_to_worker(self, mock_orchestrator, query):
        call = mock_orchestrator._keyword_routing(query)
        assert call is not None
        assert call["name"] == "analyze_skin_lesion"

    @pytest.mark.parametrize("query", [
        "What is the weather?",
        "How do I make pasta?",
        "Tell me about quantum computing",
    ])
    def test_unrelated_queries_return_none(self, mock_orchestrator, query):
        call = mock_orchestrator._keyword_routing(query)
        assert call is None


# ─── Conversation history ──────────────────────────────────────────────────────

class TestHistory:
    def test_history_grows_after_call(self, mock_orchestrator):
        mock_orchestrator.route_and_run("What is psoriasis?")
        assert len(mock_orchestrator.history) >= 2

    def test_clear_history_resets_to_empty(self, mock_orchestrator):
        mock_orchestrator.history = [{"role": "user", "content": "test"}]
        mock_orchestrator.clear_history()
        assert mock_orchestrator.history == []

    def test_history_respects_max_length(self, mock_orchestrator):
        from src.config import MAX_HISTORY_LENGTH
        for i in range(MAX_HISTORY_LENGTH + 5):
            mock_orchestrator.route_and_run(f"query {i}")
        assert len(mock_orchestrator.history) <= MAX_HISTORY_LENGTH * 2


# ─── Error resilience ──────────────────────────────────────────────────────────

class TestErrorResilience:
    def test_worker_exception_returns_safe_response(self, mock_orchestrator, dummy_image):
        with patch.object(
            mock_orchestrator.dermatology_worker, "analyze",
            side_effect=RuntimeError("inference failed")
        ):
            result = mock_orchestrator.route_and_run("check this", image=dummy_image)
        assert "response" in result
        assert result["disclaimer"]
        assert "error" in result

    def test_error_response_never_empty_disclaimer(self, mock_orchestrator, dummy_image):
        with patch.object(
            mock_orchestrator.dermatology_worker, "analyze",
            side_effect=Exception("boom")
        ):
            result = mock_orchestrator.route_and_run("test", image=dummy_image)
        assert result["disclaimer"]
