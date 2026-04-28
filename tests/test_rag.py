"""
Tests for the RAG (Retrieval-Augmented Generation) subsystem.

Covers:
  - KnowledgeRetriever: indexing, retrieval, edge cases
  - RAGWorker: initialization, answer structure, fallbacks
  - Evaluator helpers: metrics computation, synthetic sample generation

Run with:
    python -m pytest tests/test_rag.py -v
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.rag.retriever import KnowledgeRetriever, _tokenize
from src.workers.rag_worker import RAGWorker


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def retriever():
    """Real KnowledgeRetriever loaded from the project knowledge base."""
    return KnowledgeRetriever()


@pytest.fixture
def mock_rag_worker():
    """RAGWorker with Ollama mocked out."""
    with patch("src.workers.rag_worker.RAGWorker._init_ollama"):
        worker = RAGWorker(use_ollama=True)
        worker._client = MagicMock()
        mock_resp = MagicMock()
        mock_resp.message.content = (
            "Based on the knowledge base, this appears consistent with a benign condition. "
            "Please consult a dermatologist for a proper evaluation."
        )
        worker._client.chat.return_value = mock_resp
        return worker


# ─── Tokenizer ────────────────────────────────────────────────────────────────

class TestTokenizer:
    def test_lowercases_and_splits(self):
        tokens = _tokenize("Melanoma is Dangerous")
        assert "melanoma" in tokens
        assert "dangerous" in tokens

    def test_removes_stopwords(self):
        tokens = _tokenize("what is the mole")
        assert "the" not in tokens
        assert "is" not in tokens
        assert "mole" in tokens

    def test_filters_short_tokens(self):
        tokens = _tokenize("I a it go mole")
        assert "mole" in tokens
        for t in tokens:
            assert len(t) > 2

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_non_alpha_stripped(self):
        tokens = _tokenize("mole? lesion! 123 rash.")
        assert "mole" in tokens
        assert "lesion" in tokens
        assert "rash" in tokens


# ─── KnowledgeRetriever ───────────────────────────────────────────────────────

class TestKnowledgeRetriever:
    def test_loads_documents(self, retriever):
        assert retriever.doc_count > 0

    def test_retrieve_returns_list(self, retriever):
        results = retriever.retrieve("melanoma")
        assert isinstance(results, list)

    def test_retrieve_top_k_respected(self, retriever):
        results = retriever.retrieve("skin cancer", top_k=2)
        assert len(results) <= 2

    def test_retrieve_melanoma_query_finds_melanoma(self, retriever):
        results = retriever.retrieve("what is melanoma", top_k=3)
        assert len(results) > 0
        conditions = [r.get("condition") for r in results]
        assert "melanoma" in conditions

    def test_retrieve_abcde_query(self, retriever):
        results = retriever.retrieve("ABCDE rule mole check", top_k=3)
        assert len(results) > 0
        ids = [r.get("id") for r in results]
        assert "gen_abcde" in ids

    def test_retrieve_results_have_score(self, retriever):
        results = retriever.retrieve("psoriasis itchy skin")
        for r in results:
            assert "score" in r
            assert isinstance(r["score"], float)
            assert 0.0 <= r["score"] <= 1.0

    def test_retrieve_scores_descending(self, retriever):
        results = retriever.retrieve("melanoma cancer urgent", top_k=5)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_retrieve_empty_query_returns_empty(self, retriever):
        results = retriever.retrieve("")
        assert results == []

    def test_retrieve_stopwords_only_returns_empty(self, retriever):
        results = retriever.retrieve("the is a and")
        assert results == []

    def test_missing_kb_file_returns_empty(self, tmp_path):
        bad_path = tmp_path / "nonexistent.json"
        r = KnowledgeRetriever(kb_path=bad_path)
        assert r.doc_count == 0
        assert r.retrieve("melanoma") == []

    def test_custom_kb_path(self, tmp_path):
        kb = [
            {
                "id": "test_doc",
                "condition": "test",
                "title": "Test Condition",
                "text": "This is a test document about skin lesions.",
                "tags": ["test", "lesion"],
            }
        ]
        kb_file = tmp_path / "kb.json"
        kb_file.write_text(json.dumps(kb))
        r = KnowledgeRetriever(kb_path=kb_file)
        assert r.doc_count == 1
        results = r.retrieve("lesion")
        assert len(results) == 1
        assert results[0]["id"] == "test_doc"

    def test_retrieve_does_not_mutate_original_docs(self, retriever):
        results = retriever.retrieve("melanoma")
        if results:
            results[0]["score"] = 999.0
        results2 = retriever.retrieve("melanoma")
        if results2:
            assert results2[0].get("score") != 999.0


# ─── RAGWorker initialisation ─────────────────────────────────────────────────

class TestRAGWorkerInit:
    def test_worker_creates_without_crash(self, mock_rag_worker):
        assert mock_rag_worker is not None

    def test_use_ollama_flag(self, mock_rag_worker):
        assert mock_rag_worker.use_ollama is True

    def test_retriever_attached(self, mock_rag_worker):
        assert mock_rag_worker._retriever is not None
        assert mock_rag_worker._retriever.doc_count > 0


# ─── RAGWorker.answer() output structure ─────────────────────────────────────

class TestRAGWorkerAnswer:
    REQUIRED_KEYS = {"answer", "sources", "query", "context_used", "disclaimer"}

    def test_returns_all_required_keys(self, mock_rag_worker):
        result = mock_rag_worker.answer("What is melanoma?")
        assert self.REQUIRED_KEYS.issubset(result.keys())

    def test_answer_is_non_empty_string(self, mock_rag_worker):
        result = mock_rag_worker.answer("What is melanoma?")
        assert isinstance(result["answer"], str)
        assert len(result["answer"]) > 0

    def test_sources_is_list(self, mock_rag_worker):
        result = mock_rag_worker.answer("How do I check a mole?")
        assert isinstance(result["sources"], list)

    def test_disclaimer_always_present(self, mock_rag_worker):
        result = mock_rag_worker.answer("What is psoriasis?")
        assert result["disclaimer"]
        assert len(result["disclaimer"]) > 20

    def test_query_echoed(self, mock_rag_worker):
        q = "Tell me about actinic keratosis"
        result = mock_rag_worker.answer(q)
        assert result["query"] == q

    def test_sources_have_expected_fields(self, mock_rag_worker):
        result = mock_rag_worker.answer("melanoma risk factors")
        for src in result["sources"]:
            assert "title" in src
            assert "condition" in src
            assert "score" in src

    def test_disclaimer_contains_medical_warning(self, mock_rag_worker):
        result = mock_rag_worker.answer("What is eczema?")
        # disclaimer or answer must reference professional consultation
        combined = result["answer"] + result["disclaimer"]
        assert any(
            kw in combined.lower()
            for kw in ["doctor", "professional", "dermatologist"]
        )


# ─── RAGWorker fallbacks ──────────────────────────────────────────────────────

class TestRAGWorkerFallbacks:
    def test_ollama_failure_returns_template_answer(self, mock_rag_worker):
        mock_rag_worker._client.chat.side_effect = RuntimeError("connection refused")
        result = mock_rag_worker.answer("What is seborrheic keratosis?")
        assert "answer" in result
        assert result["answer"]

    def test_no_context_response_for_unrelated_query(self, mock_rag_worker):
        result = mock_rag_worker.answer("how do I make pasta carbonara")
        assert "answer" in result
        assert "disclaimer" in result

    def test_template_fallback_with_no_client(self, mock_rag_worker):
        mock_rag_worker._client = None
        mock_rag_worker.use_ollama = True
        result = mock_rag_worker.answer("What is basal cell carcinoma?")
        assert "answer" in result
        assert result["answer"]


# ─── Evaluator helpers ────────────────────────────────────────────────────────

class TestEvaluatorMetrics:
    def test_perfect_accuracy(self):
        from eval.evaluate import compute_metrics
        from src.workers.dermatology import SKIN_CLASSES

        preds = SKIN_CLASSES[:5]
        gt = SKIN_CLASSES[:5]
        metrics = compute_metrics(preds, gt, SKIN_CLASSES)
        assert metrics["accuracy"] == 1.0

    def test_zero_accuracy(self):
        from eval.evaluate import compute_metrics
        from src.workers.dermatology import SKIN_CLASSES

        preds = ["unknown"] * 5
        gt = ["melanoma"] * 5
        metrics = compute_metrics(preds, gt, SKIN_CLASSES)
        assert metrics["accuracy"] == 0.0

    def test_partial_accuracy(self):
        from eval.evaluate import compute_metrics
        from src.workers.dermatology import SKIN_CLASSES

        preds = ["melanoma", "melanoma", "benign_nevus", "benign_nevus"]
        gt = ["melanoma", "benign_nevus", "benign_nevus", "benign_nevus"]
        metrics = compute_metrics(preds, gt, SKIN_CLASSES)
        assert metrics["accuracy"] == pytest.approx(0.75)

    def test_all_metric_keys_present(self):
        from eval.evaluate import compute_metrics
        from src.workers.dermatology import SKIN_CLASSES

        metrics = compute_metrics(["melanoma"], ["melanoma"], SKIN_CLASSES)
        assert "accuracy" in metrics
        assert "macro_f1" in metrics
        assert "per_class" in metrics
        assert "confusion_matrix" in metrics

    def test_empty_predictions(self):
        from eval.evaluate import compute_metrics
        from src.workers.dermatology import SKIN_CLASSES

        metrics = compute_metrics([], [], SKIN_CLASSES)
        assert "error" in metrics

    def test_synthetic_sample_generation(self):
        from eval.evaluate import build_synthetic_samples
        from src.workers.dermatology import SKIN_CLASSES
        from PIL import Image

        samples = build_synthetic_samples(n_per_class=2)
        assert len(samples) == len(SKIN_CLASSES) * 2
        for img, label in samples:
            assert isinstance(img, Image.Image)
            assert label in SKIN_CLASSES
            assert img.size == (224, 224)

    def test_safety_adherence_rate_computed(self):
        from scripts.evaluate import compute_metrics
        from src.workers.dermatology import SKIN_CLASSES

        raw = [
            {"disclaimer": "I am not a doctor.", "plain_explanation": "test"},
            {"disclaimer": "I am not a doctor.", "plain_explanation": "test"},
            {"disclaimer": "", "plain_explanation": "no disclaimer here"},
        ]
        metrics = compute_metrics(
            ["melanoma", "benign_nevus", "unknown"],
            ["melanoma", "benign_nevus", "unknown"],
            SKIN_CLASSES,
            raw_results=raw,
        )
        assert "safety_adherence_rate" in metrics
        assert metrics["safety_adherence_rate"] == pytest.approx(2 / 3, abs=1e-3)

    def test_scripts_evaluate_mock_mode(self):
        """Smoke-test scripts/evaluate.py dry run via its public functions."""
        from scripts.evaluate import build_synthetic_samples, run_evaluation
        from unittest.mock import patch
        from src.workers.dermatology import DermatologyWorker

        samples = build_synthetic_samples(n_per_class=1)
        with patch("src.workers.dermatology.DermatologyWorker._init_ollama"):
            worker = DermatologyWorker(use_ollama=True)
        metrics = run_evaluation(samples, worker, mock=True, mock_correct_rate=1.0)
        assert metrics["accuracy"] == 1.0
        assert "macro_f1" in metrics
        assert "safety_adherence_rate" in metrics


# ─── RAGWorker v2 (src/workers/rag.py) ───────────────────────────────────────

class TestRAGWorkerV2:
    """Tests for the new domain-aware RAGWorker using derm_guidelines.json."""

    @pytest.fixture
    def mock_rag_v2(self):
        from unittest.mock import MagicMock, patch
        with patch("src.workers.rag.RAGWorker._init_ollama"):
            from src.workers.rag import RAGWorker
            w = RAGWorker(use_ollama=True)
            w._client = MagicMock()
            resp = MagicMock()
            resp.message.content = (
                "Based on the knowledge base, this appears consistent with a "
                "common skin condition. Please consult a dermatologist."
            )
            w._client.chat.return_value = resp
            return w

    def test_worker_loads_guidelines_kb(self, mock_rag_v2):
        # Guidelines KB has 45 entries; fallback has 34 — either works.
        assert mock_rag_v2.doc_count >= 34

    def test_answer_returns_required_keys(self, mock_rag_v2):
        result = mock_rag_v2.answer("What is the ABCDE rule?")
        for key in {"answer", "sources", "query", "context_used", "disclaimer"}:
            assert key in result

    def test_domain_filter_dermatology(self, mock_rag_v2):
        result = mock_rag_v2.answer("melanoma risk", domain_filter="dermatology")
        assert "answer" in result
        assert result["disclaimer"]

    def test_get_context_for_synthesis_returns_str(self, mock_rag_v2):
        ctx = mock_rag_v2.get_context_for_synthesis("melanoma dark skin", top_k=2)
        assert isinstance(ctx, str)

    def test_get_context_compact_is_shorter(self, mock_rag_v2):
        full = mock_rag_v2._format_context(
            mock_rag_v2._retriever.retrieve("melanoma", top_k=2), compact=False
        )
        compact = mock_rag_v2._format_context(
            mock_rag_v2._retriever.retrieve("melanoma", top_k=2), compact=True
        )
        assert len(compact) <= len(full)

    def test_skin_of_color_chunks_retrievable(self, mock_rag_v2):
        result = mock_rag_v2.answer("How does melanoma look on dark skin?", top_k=5)
        conditions = [s.get("condition") for s in result.get("sources", [])]
        # Either melanoma-specific or general skin-of-color chunks should appear.
        assert len(result["sources"]) > 0

    def test_fitzpatrick_chunks_retrievable(self, mock_rag_v2):
        result = mock_rag_v2.answer("Fitzpatrick scale skin types", top_k=3)
        assert len(result["sources"]) > 0

    def test_no_context_fallback(self, mock_rag_v2):
        result = mock_rag_v2.answer("how to bake bread step by step recipe pasta")
        assert "answer" in result
        assert result["disclaimer"]

    def test_ollama_failure_falls_back_gracefully(self, mock_rag_v2):
        mock_rag_v2._client.chat.side_effect = RuntimeError("timeout")
        result = mock_rag_v2.answer("What is actinic keratosis?")
        assert result["answer"]

    def test_disclaimer_present_in_all_responses(self, mock_rag_v2):
        for query in ["melanoma", "eczema treatment", "scabies rural"]:
            result = mock_rag_v2.answer(query)
            assert result["disclaimer"]
