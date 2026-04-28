"""
RAG Worker v2 for G4G RuralClinic AI.

Improvements over rag_worker.py:
  - Uses the comprehensive data/knowledge/derm_guidelines.json knowledge base
    (45 chunks covering ABCDE criteria, all skin classes, skin-of-color
     presentations, Fitzpatrick types, rural/tropical conditions, WHO referral
     criteria) with fallback to src/rag/knowledge_base.json.
  - Domain-aware retrieval: can restrict to dermatology-specific chunks.
  - Skin-of-color context: explicitly prompts Gemma 4 to consider Fitzpatrick
     type variation when synthesising answers.
  - get_context_for_synthesis(): lightweight retrieval-only path used by the
     orchestrator to enrich analysis synthesis without a full LLM round-trip.

All outputs carry safety enforcement (no diagnostic certainty, mandatory
disclaimer). Fully offline after the initial model pull.

Project: G4G RuralClinic AI — offline frontier AI for accessible
dermatology screening in rural and underserved communities.
"""

import logging
from pathlib import Path
from typing import Optional

from src.config import (
    DERM_GUIDELINES_PATH,
    MODEL_NAME,
    OFFLINE_MODE,
    OLLAMA_HOST,
    RAG_KNOWLEDGE_BASE_PATH,
    RAG_TOP_K,
    SAFETY_DISCLAIMER,
)
from src.rag.retriever import KnowledgeRetriever
from src.utils.safety import enforce_safety

logger = logging.getLogger(__name__)


# ─── Prompts ──────────────────────────────────────────────────────────────────

_RAG_SYSTEM = (
    "You are a compassionate, evidence-based AI health educator for rural and "
    "underserved communities. You answer dermatology questions using ONLY the "
    "knowledge excerpts provided. You are aware that skin conditions present "
    "differently across Fitzpatrick skin types I–VI, and you note these "
    "differences when relevant. You never make a specific diagnosis. You "
    "always recommend professional consultation for personal skin concerns."
)

_RAG_USER_TEMPLATE = """KNOWLEDGE EXCERPTS:
{context}

USER QUESTION: {question}

Instructions:
- Answer in 3–5 clear, plain-language sentences.
- If excerpts mention differences between skin tones (Fitzpatrick types), include that information.
- If the excerpts do not contain enough information, state this clearly.
- Do NOT fabricate facts not present in the excerpts.
- End by recommending consultation with a qualified dermatologist or healthcare professional."""

_CONTEXT_ONLY_TEMPLATE = """DOMAIN: {domain}

RETRIEVED KNOWLEDGE:
{context}"""


class RAGWorker:
    """
    Retrieval-Augmented Generation worker using the comprehensive dermatology
    guidelines knowledge base.

    Two modes of operation:
      answer()                   — Full Q&A: retrieve + Gemma 4 synthesis.
      get_context_for_synthesis() — Retrieval only; context string returned for
                                   use in the orchestrator's synthesis prompt.

    Both modes work offline after the initial `ollama pull`.
    """

    def __init__(
        self,
        use_ollama: bool = True,
        retriever: Optional[KnowledgeRetriever] = None,
    ):
        self.use_ollama = use_ollama
        self._client = None

        # Load knowledge base — prefer the richer guidelines file with fallback.
        if retriever is not None:
            self._retriever = retriever
        else:
            guidelines_path = Path(DERM_GUIDELINES_PATH)
            if guidelines_path.exists():
                logger.info("RAGWorker: loading guidelines KB from %s", guidelines_path)
                self._retriever = KnowledgeRetriever(kb_path=guidelines_path)
            else:
                logger.warning(
                    "RAGWorker: guidelines KB not found at %s — falling back to %s",
                    guidelines_path,
                    RAG_KNOWLEDGE_BASE_PATH,
                )
                self._retriever = KnowledgeRetriever(kb_path=Path(RAG_KNOWLEDGE_BASE_PATH))

        if use_ollama:
            self._init_ollama()

        logger.info(
            "RAGWorker ready | backend=%s | kb_docs=%d | offline=%s",
            "ollama" if self.use_ollama else "template",
            self._retriever.doc_count,
            OFFLINE_MODE,
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

    def answer(
        self,
        query: str,
        top_k: int = RAG_TOP_K,
        domain_filter: Optional[str] = None,
    ) -> dict:
        """
        Answer a dermatology knowledge question with retrieved context.

        Args:
            query:         User's question (free text).
            top_k:         Number of knowledge chunks to retrieve.
            domain_filter: If provided (e.g. "dermatology"), retrieves only
                           chunks with a matching 'domain' field. Currently
                           used for diagnostic hints; set None for general use.

        Returns:
            dict with keys:
              answer        (str)        — Safety-enforced plain-language answer.
              sources       (list[dict]) — Retrieved chunk titles/conditions/scores.
              query         (str)        — Original question echoed back.
              context_used  (str)        — Raw context fed to the model (for audit).
              disclaimer    (str)        — Medical safety disclaimer.
        """
        logger.info("RAGWorker.answer | query=%.80s | domain=%s", query, domain_filter)

        chunks = self._retriever.retrieve(query, top_k=top_k)

        # Optional domain filter: keep chunks matching the requested domain.
        if domain_filter:
            filtered = [c for c in chunks if c.get("domain") == domain_filter]
            if filtered:
                chunks = filtered

        if not chunks:
            return self._no_context_response(query)

        context = self._format_context(chunks)
        prompt = _RAG_USER_TEMPLATE.format(context=context, question=query)
        raw_answer = self._generate(prompt)
        safe_answer = enforce_safety(raw_answer)

        return {
            "answer": safe_answer,
            "sources": [
                {
                    "title": c.get("title"),
                    "condition": c.get("condition"),
                    "domain": c.get("domain"),
                    "score": round(c.get("score", 0.0), 3),
                }
                for c in chunks
            ],
            "query": query,
            "context_used": context,
            "disclaimer": SAFETY_DISCLAIMER,
        }

    def get_context_for_synthesis(
        self,
        query: str,
        top_k: int = 2,
    ) -> str:
        """
        Lightweight retrieval-only path: return a formatted context string
        for use in the orchestrator's image-analysis synthesis prompt.

        Does NOT call the LLM — pure TF-IDF retrieval only.
        Returns an empty string if no relevant chunks are found.
        """
        chunks = self._retriever.retrieve(query, top_k=top_k)
        if not chunks:
            return ""
        return self._format_context(chunks, compact=True)

    # ── Generation backends ───────────────────────────────────────────────────

    def _generate(self, user_prompt: str) -> str:
        if self.use_ollama and self._client:
            try:
                resp = self._client.chat(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": _RAG_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    options={"temperature": 0.15},
                )
                return resp.message.content
            except Exception as exc:
                logger.warning("RAGWorker: Ollama generation failed (%s) — template", exc)

        return self._template_answer(user_prompt)

    def _template_answer(self, prompt: str) -> str:
        """Direct-context fallback when the LLM is unavailable."""
        lines = [l.strip() for l in prompt.split("\n") if l.strip().startswith("[")]
        if lines:
            return (
                "From the dermatology knowledge base:\n\n"
                + "\n".join(lines[:4])
                + "\n\nPlease consult a qualified dermatologist for personal medical advice."
            )
        return (
            "I was unable to retrieve specific information. "
            "Please consult a qualified dermatologist or healthcare professional."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _format_context(chunks: list[dict], compact: bool = False) -> str:
        if compact:
            return "\n\n".join(
                f"[{c.get('condition', 'general').upper()}] {c.get('title', '')}: "
                f"{c.get('text', '')[:200]}…"
                for c in chunks
            )
        return "\n\n".join(
            f"[{c.get('condition', 'general').upper()} — {c.get('title', '')}]\n{c.get('text', '')}"
            for c in chunks
        )

    def _no_context_response(self, query: str) -> dict:
        return {
            "answer": enforce_safety(
                "I couldn't find specific information about that in the knowledge base. "
                "Please consult a qualified dermatologist or healthcare professional "
                "for accurate guidance."
            ),
            "sources": [],
            "query": query,
            "context_used": "",
            "disclaimer": SAFETY_DISCLAIMER,
        }

    @property
    def doc_count(self) -> int:
        return self._retriever.doc_count
