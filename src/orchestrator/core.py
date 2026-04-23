"""
Core orchestrator for G4G RuralClinic AI.

Responsibilities:
  1. Load Gemma 4 (via Ollama or HuggingFace pipeline).
  2. Use Gemma 4's native function-calling API to decide which specialist
     worker(s) to invoke for a given query + image.
  3. Dispatch to the chosen workers and collect structured results.
  4. Synthesise a final plain-language response with safety guardrails.
  5. Maintain conversation history and produce a traceable function-call log.

Full routing priority:
  Image present  → always analyze_skin_lesion; also retrieve_dermatology_knowledge
                   if the query looks like a knowledge question.
  Text only      → Gemma 4 native tool calling (returns 0–2 tools).
  Fallback       → keyword routing (analyze) or knowledge-query heuristic (RAG).

Project: G4G RuralClinic AI — offline frontier AI for accessible
dermatology screening in rural and underserved communities.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from PIL import Image

from src.config import (
    MAX_HISTORY_LENGTH,
    MODEL_NAME,
    OFFLINE_MODE,
    OLLAMA_HOST,
    SAFETY_DISCLAIMER,
    HF_MODEL_NAME,
)
from src.tools import TOOL_SCHEMAS
from src.utils.safety import enforce_safety
from src.workers.dermatology import DermatologyWorker
from src.workers.rag_worker import RAGWorker

logger = logging.getLogger(__name__)

# ─── System prompt ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a compassionate AI health assistant for rural and underserved communities.
Your role is to help users understand skin conditions using image analysis and symptom descriptions.

ABSOLUTE RULES — never break these:
1. You are NOT a doctor. Never make a definitive medical diagnosis.
2. Always recommend consulting a qualified dermatologist or healthcare professional.
3. When the user provides a skin image OR describes a skin symptom, call the `analyze_skin_lesion` tool.
4. When the user asks a knowledge question about skin conditions, call `retrieve_dermatology_knowledge`.
5. Explain findings in plain language — avoid jargon.
6. Be empathetic, supportive, and honest about uncertainty.
7. If you are unsure, say so clearly."""

# ─── Synthesis prompt ─────────────────────────────────────────────────────────
_SYNTHESIS_PROMPT_TEMPLATE = """The user asked: "{query}"

A specialist AI analysis produced this structured result:
  Classification : {classification}
  Confidence     : {confidence}
  Risk level     : {risk_level}
  Visual notes   : {visual_description}
  ABCDE notes    : {abcde_notes}
  Explanation    : {plain_explanation}
  Recommended    : {recommended_action}
{rag_section}
Write a compassionate, plain-language summary for the user (under 180 words).
Emphasise that this is NOT a diagnosis and they MUST consult a dermatologist.
Start with what was found, then what it might mean, then what to do next."""

# Question-like prefixes used for knowledge-query heuristic
_QUESTION_STARTS = (
    "what", "how", "why", "when", "which", "who",
    "is ", "are ", "can ", "does ", "should ", "do ",
    "tell me", "explain", "describe", "what's",
)


class Orchestrator:
    """
    Central coordinator for the RuralClinic AI system.

    Manages Gemma 4 inference, multi-tool routing, worker dispatch,
    response synthesis, and conversation history in a single stateful object.

    Typical usage::

        orch = Orchestrator()
        result = orch.route_and_run("Is this mole dangerous?", image=pil_image)
        print(result["response"])
    """

    def __init__(self, use_ollama: bool = True):
        self.use_ollama = use_ollama
        self.history: list[dict] = []
        self.function_call_log: list[dict] = []

        self.dermatology_worker = DermatologyWorker(use_ollama=use_ollama)
        self.rag_worker = RAGWorker(use_ollama=use_ollama)

        self._client = None
        self._model = None
        self._processor = None

        if use_ollama:
            self._init_ollama()
        else:
            self._init_hf_pipeline()

        logger.info(
            "Orchestrator ready | backend=%s | offline=%s",
            "ollama" if self.use_ollama else "hf",
            OFFLINE_MODE,
        )

    # ── Backend initialisation ────────────────────────────────────────────────

    def _init_ollama(self):
        try:
            import ollama
            self._client = ollama.Client(host=OLLAMA_HOST)
            available = [m.model for m in self._client.list().models]
            logger.info("Ollama available models: %s", available)
            if not any(MODEL_NAME in m for m in available):
                logger.warning(
                    "Model '%s' not found — run: ollama pull %s", MODEL_NAME, MODEL_NAME
                )
        except Exception as exc:
            logger.error("Ollama init failed (%s) — falling back to HF pipeline", exc)
            self.use_ollama = False
            self._init_hf_pipeline()

    def _init_hf_pipeline(self):
        try:
            import torch
            from transformers import AutoProcessor

            try:
                from transformers import AutoModelForImageTextToText
                _ModelClass = AutoModelForImageTextToText
            except ImportError:
                from transformers import AutoModelForCausalLM
                _ModelClass = AutoModelForCausalLM
                logger.warning(
                    "AutoModelForImageTextToText unavailable; using AutoModelForCausalLM. "
                    "Upgrade transformers>=4.50 or switch to Ollama backend."
                )

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if device == "cuda" else torch.float32

            logger.info("Loading HF model %s on %s…", HF_MODEL_NAME, device)
            self._processor = AutoProcessor.from_pretrained(
                HF_MODEL_NAME, local_files_only=OFFLINE_MODE
            )
            self._model = _ModelClass.from_pretrained(
                HF_MODEL_NAME,
                torch_dtype=dtype,
                device_map=device,
                local_files_only=OFFLINE_MODE,
            )
            logger.info("HF model loaded on %s", device)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Gemma 4. Ensure Ollama is running or the HF model is "
                f"downloaded offline. Original error: {exc}"
            ) from exc

    # ── Main public interface ─────────────────────────────────────────────────

    def route_and_run(
        self, query: str, image: Optional[Image.Image] = None
    ) -> dict:
        """
        Primary entry point: route a query + optional image through multi-tool
        dispatch and return a synthesised plain-language response.

        Args:
            query: User's text question or symptom description.
            image: Optional PIL Image of a skin lesion.

        Returns:
            dict with keys:
              response        (str)        — Final synthesised answer.
              worker_result   (dict|None)  — Raw output from DermatologyWorker.
              rag_result      (dict|None)  — Raw output from RAGWorker.
              function_calls  (list)       — Audit trail of all tool calls.
              disclaimer      (str)        — Medical safety disclaimer.
              timestamp       (str)        — UTC ISO-8601 timestamp.
              error           (str)        — Present only if an exception occurred.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info("route_and_run | has_image=%s | query=%.80s", image is not None, query)
        self.function_call_log.clear()

        try:
            tool_calls = self._decide_tool_calls(query, image)

            worker_result: Optional[dict] = None
            rag_result: Optional[dict] = None

            for tc in tool_calls:
                name = tc["name"]
                args = tc.get("arguments", {})

                if name == "analyze_skin_lesion":
                    symptoms = args.get("symptoms", query)
                    worker_result = self.dermatology_worker.analyze(
                        image=image, symptoms=symptoms
                    )
                    self._log_function_call(name, args, worker_result)

                elif name == "retrieve_dermatology_knowledge":
                    knowledge_query = args.get("query", query)
                    rag_result = self.rag_worker.answer(knowledge_query)
                    self._log_function_call(name, args, rag_result)

            response = self._synthesise_response(query, worker_result, rag_result)
            response = enforce_safety(response)

            self._update_history(query, response)

            return {
                "response": response,
                "worker_result": worker_result,
                "rag_result": rag_result,
                "function_calls": list(self.function_call_log),
                "disclaimer": SAFETY_DISCLAIMER,
                "timestamp": timestamp,
            }

        except Exception as exc:
            logger.exception("route_and_run error: %s", exc)
            return {
                "response": (
                    "I encountered an error processing your request. "
                    "Please try again, or consult a healthcare professional directly."
                ),
                "worker_result": None,
                "rag_result": None,
                "function_calls": list(self.function_call_log),
                "disclaimer": SAFETY_DISCLAIMER,
                "timestamp": timestamp,
                "error": str(exc),
            }

    # ── Multi-tool routing ────────────────────────────────────────────────────

    def _decide_tool_calls(
        self, query: str, image: Optional[Image.Image]
    ) -> list[dict]:
        """
        Determine which tools to call for this query + image combination.

        Priority:
          1. Image present → analyze_skin_lesion always.
             Also add retrieve_dermatology_knowledge if query is question-like.
          2. No image, Ollama available → let Gemma 4 decide (may return 0–2 tools).
          3. No image, no Ollama → keyword/heuristic fallback.
        """
        if image is not None:
            calls: list[dict] = [
                {"name": "analyze_skin_lesion", "arguments": {"symptoms": query}}
            ]
            if self._is_knowledge_query(query):
                calls.append(
                    {"name": "retrieve_dermatology_knowledge", "arguments": {"query": query}}
                )
            return calls

        if self.use_ollama and self._client:
            return self._ollama_tool_calls(query)

        return self._keyword_fallback(query)

    def _ollama_tool_calls(self, query: str) -> list[dict]:
        """
        Use Gemma 4 native function calling to select 0, 1, or 2 tools.
        Falls back to keyword routing if Ollama raises an exception.
        """
        try:
            messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
            messages.extend(self.history[-MAX_HISTORY_LENGTH * 2:])
            messages.append({"role": "user", "content": query})

            response = self._client.chat(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOL_SCHEMAS,
                options={"temperature": 0.1},
            )
            if hasattr(response, "message") and response.message.tool_calls:
                calls = []
                for tc in response.message.tool_calls:
                    logger.info("Gemma 4 selected tool: %s", tc.function.name)
                    calls.append({
                        "name": tc.function.name,
                        "arguments": tc.function.arguments or {},
                    })
                return calls

            logger.info("Gemma 4 selected no tool — general response")
            return []

        except Exception as exc:
            logger.warning("Ollama tool decision failed (%s) — keyword fallback", exc)
            return self._keyword_fallback(query)

    def _keyword_fallback(self, query: str) -> list[dict]:
        """
        Heuristic routing used when Ollama is unavailable.
        Knowledge questions → RAG; symptom descriptions → analysis.
        """
        single = self._keyword_routing(query)
        if single is None:
            return []
        # Override to RAG when the matched query is question-like
        if self._is_knowledge_query(query):
            return [{"name": "retrieve_dermatology_knowledge", "arguments": {"query": query}}]
        return [single]

    # ── Legacy single-call routing (kept for backward compatibility) ──────────

    def _keyword_routing(self, query: str) -> Optional[dict]:
        """
        Lightweight keyword-based routing to analyze_skin_lesion.
        Returns a single tool-call dict, or None for unrelated queries.

        Note: _decide_tool_calls() wraps this and may reroute to RAG for
        question-like queries. Direct callers (e.g. tests) see the raw result.
        """
        derm_keywords = {
            "skin", "mole", "lesion", "rash", "melanoma", "nevus", "spot",
            "bump", "growth", "itch", "dermat", "freckle", "wart", "acne",
            "psoriasis", "eczema", "blister", "sore", "patch", "blotch",
            "discolor", "pigment", "lump", "crust", "scale",
        }
        if any(kw in query.lower() for kw in derm_keywords):
            logger.info("Keyword routing → analyze_skin_lesion")
            return {"name": "analyze_skin_lesion", "arguments": {"symptoms": query}}
        return None

    # ── Response synthesis ────────────────────────────────────────────────────

    def _synthesise_response(
        self,
        query: str,
        worker_result: Optional[dict],
        rag_result: Optional[dict] = None,
    ) -> str:
        """
        Produce a plain-language response from one or both worker outputs.

        - No workers called → general conversational response.
        - RAG only → use RAG answer directly.
        - Analysis (± RAG) → Gemma 4 synthesises from structured result,
          optionally enriched with retrieved knowledge context.
        """
        if worker_result is None and rag_result is None:
            return self._general_response(query)

        if worker_result is None and rag_result is not None:
            return rag_result.get("answer", "")

        # worker_result present — build synthesis prompt
        rag_section = ""
        if rag_result and rag_result.get("answer"):
            rag_section = (
                f"\nAdditional knowledge context:\n  {rag_result['answer']}\n"
            )

        prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(
            query=query,
            classification=worker_result.get("classification", "unknown"),
            confidence=worker_result.get("confidence", "unknown"),
            risk_level=worker_result.get("risk_level", "unknown"),
            visual_description=worker_result.get("visual_description", "N/A"),
            abcde_notes=worker_result.get("abcde_notes", "N/A"),
            plain_explanation=worker_result.get("plain_explanation", "N/A"),
            recommended_action=worker_result.get("recommended_action", "N/A"),
            rag_section=rag_section,
        )

        if self.use_ollama and self._client:
            try:
                resp = self._client.chat(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.3},
                )
                return resp.message.content
            except Exception as exc:
                logger.warning("Ollama synthesis failed (%s) — template", exc)

        return self._template_synthesis(worker_result)

    def _general_response(self, query: str) -> str:
        if self.use_ollama and self._client:
            try:
                messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
                messages.extend(self.history[-MAX_HISTORY_LENGTH * 2:])
                messages.append({"role": "user", "content": query})
                resp = self._client.chat(
                    model=MODEL_NAME,
                    messages=messages,
                    options={"temperature": 0.4},
                )
                return resp.message.content
            except Exception as exc:
                logger.warning("Ollama general response failed: %s", exc)

        return (
            "I can help with skin-related questions. Please describe your symptoms "
            "or upload a photo of the area of concern. Remember, I provide "
            "informational guidance only — always consult a healthcare professional "
            "for medical concerns."
        )

    def _template_synthesis(self, result: dict) -> str:
        cls = result.get("classification", "unclassified").replace("_", " ").title()
        conf = result.get("confidence", "unknown")
        risk = result.get("risk_level", "unknown")
        explanation = result.get("plain_explanation", "No explanation available.")
        action = result.get("recommended_action", "Please consult a dermatologist.")
        visual = result.get("visual_description", "")

        parts = [
            f"The image analysis suggests: **{cls}** "
            f"(confidence: {conf}, risk level: {risk}).",
            "",
            explanation,
        ]
        if visual:
            parts += ["", f"Visual observations: {visual}"]
        parts += ["", f"Recommended next step: {action}"]
        return "\n".join(parts)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_knowledge_query(text: str) -> bool:
        """Heuristic: does this text look like an informational question?"""
        q = text.lower().strip()
        return q.endswith("?") or any(q.startswith(s) for s in _QUESTION_STARTS)

    def _log_function_call(self, name: str, arguments: dict, result: dict):
        entry = {
            "tool": name,
            "arguments": arguments,
            "result_keys": list(result.keys()) if result else [],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.function_call_log.append(entry)
        logger.info(
            "FUNCTION_CALL | tool=%s | arg_keys=%s | result_keys=%s",
            name,
            list(arguments.keys()),
            entry["result_keys"],
        )

    def _update_history(self, query: str, response: str):
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})
        max_entries = MAX_HISTORY_LENGTH * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    def clear_history(self):
        self.history.clear()
        logger.info("Conversation history cleared")
