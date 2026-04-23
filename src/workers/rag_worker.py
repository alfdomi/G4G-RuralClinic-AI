"""
RAG (Retrieval-Augmented Generation) worker for G4G RuralClinic AI.

Answers general dermatology knowledge questions by:
  1. Retrieving relevant chunks from the offline knowledge base via TF-IDF.
  2. Passing retrieved context + question to Gemma 4 for a grounded answer.

Handles text-only queries that don't need image analysis — e.g.:
  "What is melanoma?", "How do I check a mole?", "What causes psoriasis?"

Works fully offline after the initial `ollama pull`.
"""

import logging
from typing import Optional

from src.config import MODEL_NAME, OFFLINE_MODE, OLLAMA_HOST, SAFETY_DISCLAIMER
from src.rag.retriever import KnowledgeRetriever
from src.utils.safety import enforce_safety

logger = logging.getLogger(__name__)

_RAG_PROMPT = """You are a compassionate AI health educator for rural and underserved communities.
Answer the following question using ONLY the knowledge excerpts provided below.
If the excerpts do not contain sufficient information, say so clearly — do not invent facts.
Always recommend consulting a qualified dermatologist or healthcare professional for personal concerns.

KNOWLEDGE EXCERPTS:
{context}

USER QUESTION: {question}

Provide a clear, plain-language answer in 2–4 sentences.
Do NOT make specific diagnoses. Do NOT claim certainty. Use hedging language where appropriate.
End by recommending professional consultation for any personal skin concern."""


class RAGWorker:
    """
    Retrieval-Augmented Generation worker for dermatology knowledge Q&A.

    Combines an offline TF-IDF knowledge retriever with Gemma 4 (via Ollama
    or HuggingFace) to produce grounded, safety-checked answers to general
    dermatology questions.  No internet is required after the initial model pull.
    """

    def __init__(
        self,
        use_ollama: bool = True,
        retriever: Optional[KnowledgeRetriever] = None,
    ):
        self.use_ollama = use_ollama
        self._client = None
        self._retriever = retriever if retriever is not None else KnowledgeRetriever()

        if use_ollama:
            self._init_ollama()

        logger.info(
            "RAGWorker ready | backend=%s | kb_docs=%d",
            "ollama" if self.use_ollama else "template",
            self._retriever.doc_count,
        )

    # ── Backend init ──────────────────────────────────────────────────────────

    def _init_ollama(self) -> None:
        try:
            import ollama
            self._client = ollama.Client(host=OLLAMA_HOST)
            logger.info("RAGWorker: Ollama client ready at %s", OLLAMA_HOST)
        except Exception as exc:
            logger.error("RAGWorker: Ollama init failed (%s) — template fallback", exc)
            self.use_ollama = False

    # ── Public API ────────────────────────────────────────────────────────────

    def answer(self, query: str, top_k: int = 3) -> dict:
        """
        Answer a dermatology knowledge question using retrieved context.

        Args:
            query:  User's free-text question.
            top_k:  Number of knowledge chunks to retrieve.

        Returns:
            dict with keys:
              answer        (str)        — Safety-enforced plain-language answer.
              sources       (list[dict]) — Retrieved chunk metadata (title, condition, score).
              query         (str)        — Original question echoed back.
              context_used  (str)        — Raw retrieved context fed to the model.
              disclaimer    (str)        — Medical safety disclaimer.
        """
        logger.info("RAGWorker.answer | query=%.80s", query)

        chunks = self._retriever.retrieve(query, top_k=top_k)

        if not chunks:
            return self._no_context_response(query)

        context = "\n\n".join(
            f"[{c.get('condition', 'general').upper()}] {c.get('title', '')}\n{c.get('text', '')}"
            for c in chunks
        )

        prompt = _RAG_PROMPT.format(context=context, question=query)
        raw_answer = self._generate(prompt)
        safe_answer = enforce_safety(raw_answer)

        return {
            "answer": safe_answer,
            "sources": [
                {
                    "title": c.get("title"),
                    "condition": c.get("condition"),
                    "score": round(c.get("score", 0.0), 3),
                }
                for c in chunks
            ],
            "query": query,
            "context_used": context,
            "disclaimer": SAFETY_DISCLAIMER,
        }

    # ── Generation backends ───────────────────────────────────────────────────

    def _generate(self, prompt: str) -> str:
        if self.use_ollama and self._client:
            try:
                resp = self._client.chat(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    options={"temperature": 0.2},
                )
                return resp.message.content
            except Exception as exc:
                logger.warning("RAGWorker: Ollama generation failed (%s) — template", exc)

        return self._template_answer(prompt)

    def _template_answer(self, prompt: str) -> str:
        """Structured-context fallback when the LLM is unavailable."""
        lines = [l.strip() for l in prompt.split("\n") if l.strip().startswith("[")]
        if lines:
            return (
                "Based on the knowledge base:\n\n"
                + "\n".join(lines[:3])
                + "\n\nPlease consult a qualified dermatologist for personal medical advice."
            )
        return (
            "Please consult a qualified dermatologist or healthcare professional "
            "for information about skin conditions."
        )

    def _no_context_response(self, query: str) -> dict:
        return {
            "answer": enforce_safety(
                "I couldn't find specific information about that in my knowledge base. "
                "Please consult a qualified dermatologist or healthcare professional "
                "for accurate guidance on skin conditions."
            ),
            "sources": [],
            "query": query,
            "context_used": "",
            "disclaimer": SAFETY_DISCLAIMER,
        }
