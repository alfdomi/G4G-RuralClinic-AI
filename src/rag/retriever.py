"""
Offline TF-IDF knowledge retriever for G4G RuralClinic AI.

Loads the dermatology knowledge base JSON and builds an in-memory
TF-IDF index using only numpy — no network or third-party ML libraries.
Retrieval is a cosine similarity lookup over the term-document matrix.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import RAG_KNOWLEDGE_BASE_PATH, RAG_TOP_K

logger = logging.getLogger(__name__)

_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "this", "that", "these",
    "those", "it", "its", "i", "you", "he", "she", "we", "they", "in",
    "on", "at", "by", "for", "with", "about", "as", "into", "through",
    "to", "of", "and", "or", "but", "not", "from", "if", "then", "than",
    "so", "also", "such", "your", "their", "our", "my", "his", "her",
})


def _tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z]+", text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 2]


class KnowledgeRetriever:
    """
    Lightweight TF-IDF retriever over the offline dermatology knowledge base.

    Fully offline — requires only numpy, which is already a project dependency.
    Initialises gracefully with an empty index if the knowledge base file is
    missing, so the orchestrator can always instantiate it safely.
    """

    def __init__(self, kb_path: Optional[Path] = None):
        path = Path(kb_path or RAG_KNOWLEDGE_BASE_PATH)
        if not path.exists():
            logger.warning("Knowledge base not found at %s — RAG retrieval disabled", path)
            self._docs: list[dict] = []
            self._tfidf = None
            self._vocab: dict[str, int] = {}
            return

        with open(path, "r", encoding="utf-8") as fh:
            self._docs = json.load(fh)

        self._vocab = {}
        self._tfidf = None
        self._build_index()
        logger.info("KnowledgeRetriever: indexed %d documents", len(self._docs))

    # ── Index construction ────────────────────────────────────────────────────

    def _build_index(self) -> None:
        """Build a TF-IDF term-document matrix from the knowledge base."""
        corpus = [
            " ".join([
                d.get("condition", ""),
                d.get("title", ""),
                d.get("text", ""),
                " ".join(d.get("tags", [])),
            ])
            for d in self._docs
        ]
        tokenized = [_tokenize(doc) for doc in corpus]

        vocab = sorted({t for doc in tokenized for t in doc})
        self._vocab = {t: i for i, t in enumerate(vocab)}
        V, N = len(vocab), len(tokenized)

        tf = np.zeros((N, V), dtype=np.float32)
        for i, doc in enumerate(tokenized):
            for term in doc:
                if term in self._vocab:
                    tf[i, self._vocab[term]] += 1
            total = tf[i].sum()
            if total > 0:
                tf[i] /= total

        df = (tf > 0).sum(axis=0).astype(np.float32)
        idf = np.log((N + 1) / (df + 1)) + 1.0

        tfidf = tf * idf
        norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._tfidf = tfidf / norms

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = RAG_TOP_K) -> list[dict]:
        """
        Return the top_k most relevant knowledge chunks for a query.

        Returns an empty list if the index is unavailable or no tokens match.
        Each returned dict is a copy of the original document entry plus a
        float 'score' key (cosine similarity, 0–1).
        """
        if self._tfidf is None or not self._vocab:
            return []

        tokens = _tokenize(query)
        q_vec = np.zeros(len(self._vocab), dtype=np.float32)
        for t in tokens:
            if t in self._vocab:
                q_vec[self._vocab[t]] += 1

        norm = float(np.linalg.norm(q_vec))
        if norm == 0:
            return []
        q_vec /= norm

        scores = self._tfidf @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc = dict(self._docs[idx])
                doc["score"] = float(scores[idx])
                results.append(doc)
        return results

    @property
    def doc_count(self) -> int:
        return len(self._docs)
