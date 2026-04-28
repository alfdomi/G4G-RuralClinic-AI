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
    Orchestrator with all Ollama backends mocked so no local server is needed.
    The chat() mock returns a general (no-tool-call) response by default.
    RAGWorker v2 (src/workers/rag.py) is also patched.
    """
    with patch("src.orchestrator.core.Orchestrator._init_ollama"):
        with patch("src.workers.dermatology.DermatologyWorker._init_ollama"):
            with patch("src.workers.rag.RAGWorker._init_ollama"):
                from src.orchestrator.core import Orchestrator

                orch = Orchestrator(use_ollama=True)
                orch._client = MagicMock()

                # Default: model returns no tool call → heuristic routing + general response
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


# ─── Phase 1 routing decision ─────────────────────────────────────────────────

class TestRoutingDecision:
    """Tests for the structured routing_decision output from _triage_query."""

    REQUIRED_ROUTING_KEYS = {
        "domain", "triage_level", "extracted_symptoms",
        "recommended_workers", "reasoning", "routing_source",
    }

    def test_routing_decision_in_output(self, mock_orchestrator):
        result = mock_orchestrator.route_and_run("What is melanoma?")
        assert "routing_decision" in result

    def test_routing_decision_has_required_fields(self, mock_orchestrator):
        result = mock_orchestrator.route_and_run("What is melanoma?")
        rd = result["routing_decision"]
        assert self.REQUIRED_ROUTING_KEYS.issubset(rd.keys())

    def test_image_routes_to_dermatology_domain(self, mock_orchestrator, dummy_image):
        result = mock_orchestrator.route_and_run("check this", image=dummy_image)
        assert result["routing_decision"]["domain"] == "dermatology"

    def test_image_recommends_analyze_worker(self, mock_orchestrator, dummy_image):
        with patch.object(
            mock_orchestrator.dermatology_worker, "analyze",
            return_value=_worker_result_fixture()
        ):
            result = mock_orchestrator.route_and_run("analyse spot", image=dummy_image)
        assert "analyze_skin_lesion" in result["routing_decision"]["recommended_workers"]

    def test_unrelated_query_routes_to_general(self, mock_orchestrator):
        result = mock_orchestrator.route_and_run("What is the weather today?")
        assert result["routing_decision"]["domain"] == "general"

    def test_unrelated_query_has_no_workers(self, mock_orchestrator):
        result = mock_orchestrator.route_and_run("What is the weather today?")
        assert result["routing_decision"]["recommended_workers"] == []

    def test_derm_keyword_symptom_routes_to_dermatology(self, mock_orchestrator):
        result = mock_orchestrator.route_and_run("I have a rash on my arm")
        assert result["routing_decision"]["domain"] == "dermatology"

    def test_urgent_keywords_elevate_triage(self, mock_orchestrator, dummy_image):
        with patch.object(
            mock_orchestrator.dermatology_worker, "analyze",
            return_value=_worker_result_fixture()
        ):
            result = mock_orchestrator.route_and_run(
                "this melanoma is bleeding rapidly", image=dummy_image
            )
        assert result["routing_decision"]["triage_level"] in {"urgent", "moderate"}

    def test_routing_source_is_heuristic_when_ollama_mocked(self, mock_orchestrator):
        # Mocked _client returns tool_calls=None → heuristic fallback
        result = mock_orchestrator.route_and_run("check my mole", image=None)
        assert result["routing_decision"]["routing_source"] == "heuristic"

    def test_extracted_symptoms_is_list(self, mock_orchestrator, dummy_image):
        with patch.object(
            mock_orchestrator.dermatology_worker, "analyze",
            return_value=_worker_result_fixture()
        ):
            result = mock_orchestrator.route_and_run("itchy mole", image=dummy_image)
        assert isinstance(result["routing_decision"]["extracted_symptoms"], list)

    def test_reasoning_is_non_empty_string(self, mock_orchestrator):
        result = mock_orchestrator.route_and_run("dark lesion on my back")
        rd = result["routing_decision"]
        assert isinstance(rd["reasoning"], str)
        assert len(rd["reasoning"]) > 0

    def test_error_path_includes_routing_decision(self, mock_orchestrator, dummy_image):
        with patch.object(
            mock_orchestrator.dermatology_worker, "analyze",
            side_effect=RuntimeError("crash")
        ):
            result = mock_orchestrator.route_and_run("check this", image=dummy_image)
        assert "routing_decision" in result
        assert "domain" in result["routing_decision"]


# ─── Heuristic triage ─────────────────────────────────────────────────────────

class TestHeuristicTriage:
    """Unit tests for _heuristic_triage() directly."""

    def test_image_always_dermatology(self, mock_orchestrator, dummy_image):
        decision = mock_orchestrator._heuristic_triage("anything", dummy_image)
        assert decision["domain"] == "dermatology"

    def test_image_always_recommends_analyze(self, mock_orchestrator, dummy_image):
        decision = mock_orchestrator._heuristic_triage("anything", dummy_image)
        assert "analyze_skin_lesion" in decision["recommended_workers"]

    def test_no_image_no_keyword_general(self, mock_orchestrator):
        decision = mock_orchestrator._heuristic_triage("what is the weather?", None)
        assert decision["domain"] == "general"
        assert decision["recommended_workers"] == []

    def test_derm_symptom_not_question_gets_analyze(self, mock_orchestrator):
        decision = mock_orchestrator._heuristic_triage("I have a rash on my leg", None)
        assert "analyze_skin_lesion" in decision["recommended_workers"]

    def test_derm_question_gets_rag(self, mock_orchestrator):
        decision = mock_orchestrator._heuristic_triage("What is melanoma?", None)
        assert "retrieve_dermatology_knowledge" in decision["recommended_workers"]

    def test_urgent_keyword_elevates_triage(self, mock_orchestrator):
        decision = mock_orchestrator._heuristic_triage("mole is bleeding rapidly", None)
        assert decision["triage_level"] == "urgent"

    def test_derm_keyword_without_urgent_is_moderate(self, mock_orchestrator):
        decision = mock_orchestrator._heuristic_triage("I have a mole on my back", None)
        assert decision["triage_level"] == "moderate"

    def test_general_query_is_low_triage(self, mock_orchestrator):
        decision = mock_orchestrator._heuristic_triage("hello how are you", None)
        assert decision["triage_level"] == "low"


# ─── RAGWorker v2 integration (via orchestrator) ──────────────────────────────

class TestRAGWorkerV2Integration:
    """Smoke tests for src/workers/rag.py loaded through the orchestrator."""

    def test_rag_worker_attached(self, mock_orchestrator):
        from src.workers.rag import RAGWorker
        assert isinstance(mock_orchestrator.rag_worker, RAGWorker)

    def test_rag_worker_has_docs(self, mock_orchestrator):
        # Guidelines KB (45 chunks) or fallback KB should be loaded
        assert mock_orchestrator.rag_worker.doc_count > 0

    def test_get_context_for_synthesis_returns_string(self, mock_orchestrator):
        ctx = mock_orchestrator.rag_worker.get_context_for_synthesis("melanoma", top_k=2)
        assert isinstance(ctx, str)
