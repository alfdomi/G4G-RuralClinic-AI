"""
Core orchestrator for G4G RuralClinic AI.

Two-phase agentic workflow:

  Phase 1 — Triage & Route  (_triage_query)
  ──────────────────────────────────────────
  Gemma 4 is invoked with the ROUTING_TOOL schema and returns a single
  structured JSON routing decision:
      {
        "domain": "dermatology" | "general" | "unclear",
        "triage_level": "urgent" | "moderate" | "low" | "unclear",
        "extracted_symptoms": ["itchy", "dark mole", ...],
        "recommended_workers": ["analyze_skin_lesion", ...],
        "reasoning": "ABCDE features present: asymmetric border and …"
      }
  If Gemma 4 is unavailable, _heuristic_triage() produces the same
  structure via keyword analysis.

  Phase 2 — Worker Dispatch  (_dispatch_workers)
  ────────────────────────────────────────────────
  Each worker in routing_decision["recommended_workers"] is invoked:
    analyze_skin_lesion         → DermatologyWorker (image + ABCDE)
    retrieve_dermatology_knowledge → RAGWorker v2 (guidelines KB)
  Results are logged to the function_call_log for UI transparency.

  Phase 3 — Synthesis  (_synthesise_response)
  ─────────────────────────────────────────────
  Gemma 4 produces a compassionate plain-language summary from the
  structured worker outputs, enriched with retrieved knowledge context
  and the routing triage level as editorial tone guidance.
  Safety enforcement (disclaimer + language softening) is applied last.

Conversation history is maintained across calls; clear_history() resets it.

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
from src.tools import ROUTING_SCHEMAS, TOOL_SCHEMAS
from src.utils.safety import enforce_safety
from src.workers.dermatology import DermatologyWorker
from src.workers.rag import RAGWorker

logger = logging.getLogger(__name__)

# ─── System prompts ───────────────────────────────────────────────────────────

_TRIAGE_SYSTEM = """You are a clinical triage AI for a rural dermatology screening tool.
Your task is to analyse the user's message and any attached image and call the
triage_and_route tool with a structured routing decision.

ALWAYS call triage_and_route — do not respond in plain text.

Triage rules:
- Image present OR personal skin symptoms described → domain = dermatology
- ABCDE warning signs, bleeding, rapid growth, immunocompromised → triage = urgent
- Changing mole, chronic rash, educational question about a condition → triage = moderate
- Stable lesion, stable skin concern, pure knowledge question → triage = low
- No skin involvement at all → domain = general, recommended_workers = []"""

_SYNTHESIS_SYSTEM = """You are a compassionate AI health assistant for rural and underserved communities.
You are NOT a doctor. You never make definitive diagnoses.
Always recommend consulting a qualified dermatologist or healthcare professional.
Explain findings in plain, empathetic language. Acknowledge uncertainty honestly."""

_SYNTHESIS_TEMPLATE = """TRIAGE CONTEXT:
  Domain       : {domain}
  Urgency      : {triage_level}
  Symptoms     : {symptoms}
  Reasoning    : {reasoning}

SPECIALIST ANALYSIS:
  Classification : {classification}
  Confidence     : {confidence}
  Risk level     : {risk_level}
  ABCDE notes    : {abcde_notes}
  Explanation    : {plain_explanation}
  Recommended    : {recommended_action}
{rag_section}
USER'S QUESTION: "{query}"

Instructions (follow all):
- Write a compassionate, plain-language response in 3–5 sentences (under 200 words).
- Lead with what was found and the urgency level.
- If RELEVANT KNOWLEDGE CONTEXT is provided above, use it to add clinical grounding
  (e.g. mention how conditions differ by skin tone, cite next-step timing).
- Explain what the finding might mean in simple terms; avoid jargon.
- State clearly what the person should do next (timing, who to contact).
- End by emphasising this is NOT a diagnosis and they MUST consult a clinician.
- If triage_level is 'urgent', state explicitly they should seek care today or within days."""

# Question-like prefixes used by the heuristic triage.
_QUESTION_STARTS = (
    "what", "how", "why", "when", "which", "who",
    "is ", "are ", "can ", "does ", "should ", "do ",
    "tell me", "explain", "describe", "what's",
)

# Keywords indicating dermatological concern.
_DERM_KEYWORDS = frozenset({
    "skin", "mole", "lesion", "rash", "melanoma", "nevus", "nevus", "spot",
    "bump", "growth", "itch", "dermat", "freckle", "wart", "acne",
    "psoriasis", "eczema", "blister", "sore", "patch", "blotch",
    "discolor", "pigment", "lump", "crust", "scale", "seborrheic",
    "keratosis", "carcinoma", "basal", "squamous", "vascular", "hemangioma",
    "angioma", "tinea", "fungal", "scabies", "impetigo", "abscess", "cyst",
})

# Keywords elevating triage to urgent.
_URGENT_KEYWORDS = frozenset({
    "melanoma", "cancer", "bleed", "bleeding", "ulcer", "spreading",
    "rapid", "rapidly", "emergency", "urgent", "severe", "growing fast",
    "infected", "pus", "fever", "pain",
})


class Orchestrator:
    """
    Central coordinator for the RuralClinic AI system.

    Manages the two-phase triage-then-dispatch agentic workflow, Gemma 4
    inference, worker coordination, response synthesis, and conversation
    history in a single stateful object.

    Typical usage::

        orch = Orchestrator()
        result = orch.route_and_run("Is this mole dangerous?", image=pil_image)
        print(result["response"])
        print(result["routing_decision"]["triage_level"])
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
        Primary entry point: run the two-phase agentic workflow.

        Phase 1: _triage_query() → structured routing_decision JSON.
        Phase 2: dispatch recommended workers → worker_result + rag_result.
        Phase 3: synthesise → plain-language response with safety guardrails.

        Args:
            query: User's text question or symptom description.
            image: Optional PIL Image of a skin lesion.

        Returns:
            dict with keys:
              response          (str)        — Final synthesised answer.
              routing_decision  (dict)       — Structured triage/routing JSON.
              worker_result     (dict|None)  — DermatologyWorker output.
              rag_result        (dict|None)  — RAGWorker output.
              function_calls    (list)       — Audit trail of all tool calls.
              disclaimer        (str)        — Medical safety disclaimer.
              timestamp         (str)        — UTC ISO-8601 timestamp.
              error             (str)        — Present only if an exception occurred.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        logger.info("route_and_run | has_image=%s | query=%.80s", image is not None, query)
        self.function_call_log.clear()

        try:
            # ── Phase 1: Triage & Route ───────────────────────────────────
            routing_decision = self._triage_query(query, image)
            logger.info(
                "Routing decision | domain=%s | triage=%s | workers=%s | source=%s",
                routing_decision.get("domain"),
                routing_decision.get("triage_level"),
                routing_decision.get("recommended_workers"),
                routing_decision.get("routing_source", "unknown"),
            )

            # ── Phase 2: Worker Dispatch ──────────────────────────────────
            worker_result: Optional[dict] = None
            rag_result: Optional[dict] = None

            for worker_name in routing_decision.get("recommended_workers", []):
                if worker_name == "analyze_skin_lesion":
                    symptoms = " ".join(routing_decision.get("extracted_symptoms", []))
                    symptoms = symptoms or query
                    worker_result = self.dermatology_worker.analyze(
                        image=image, symptoms=symptoms
                    )
                    self._log_function_call(
                        "analyze_skin_lesion",
                        {"symptoms": symptoms[:120]},
                        worker_result,
                    )

                elif worker_name == "retrieve_dermatology_knowledge":
                    rag_result = self.rag_worker.answer(
                        query,
                        domain_filter="dermatology"
                        if routing_decision.get("domain") == "dermatology"
                        else None,
                    )
                    self._log_function_call(
                        "retrieve_dermatology_knowledge",
                        {"query": query[:120]},
                        rag_result,
                    )

            # ── Phase 3: Synthesis ────────────────────────────────────────
            response = self._synthesise_response(
                query, routing_decision, worker_result, rag_result
            )
            response = enforce_safety(response)

            self._update_history(query, response)

            return {
                "response": response,
                "routing_decision": routing_decision,
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
                "routing_decision": {
                    "domain": "unclear",
                    "triage_level": "unclear",
                    "extracted_symptoms": [],
                    "recommended_workers": [],
                    "reasoning": "Error during processing.",
                    "routing_source": "error",
                },
                "worker_result": None,
                "rag_result": None,
                "function_calls": list(self.function_call_log),
                "disclaimer": SAFETY_DISCLAIMER,
                "timestamp": timestamp,
                "error": str(exc),
            }

    # ── Phase 1: Triage ───────────────────────────────────────────────────────

    def _triage_query(
        self, query: str, image: Optional[Image.Image]
    ) -> dict:
        """
        Produce the Phase 1 structured routing decision.

        For image inputs, uses a fast heuristic (always dermatology) to avoid
        an extra Ollama round-trip for the triage call.
        For text-only inputs, tries Gemma 4 native function calling first,
        then falls back to keyword-based heuristic.
        """
        # Fast path: image always means dermatology — skip LLM triage.
        if image is not None:
            return self._heuristic_triage(query, image)

        if self.use_ollama and self._client:
            decision = self._ollama_triage(query)
            if decision is not None:
                return decision

        return self._heuristic_triage(query, image)

    def _ollama_triage(self, query: str) -> Optional[dict]:
        """
        Ask Gemma 4 to produce a structured routing decision via ROUTING_TOOL.

        Returns the parsed routing decision dict on success, or None if Gemma 4
        does not return a tool call (falls back to heuristic caller).
        """
        try:
            messages = [{"role": "system", "content": _TRIAGE_SYSTEM}]
            messages.extend(self.history[-MAX_HISTORY_LENGTH * 2:])
            messages.append({"role": "user", "content": query})

            response = self._client.chat(
                model=MODEL_NAME,
                messages=messages,
                tools=ROUTING_SCHEMAS,
                options={"temperature": 0.05},  # Low temp for consistent routing
            )

            if (
                hasattr(response, "message")
                and response.message.tool_calls
                and response.message.tool_calls[0].function.name == "triage_and_route"
            ):
                tc = response.message.tool_calls[0]
                args = dict(tc.function.arguments or {})
                args["routing_source"] = "gemma4_function_call"
                logger.info(
                    "Gemma 4 triage | domain=%s | triage=%s | workers=%s",
                    args.get("domain"),
                    args.get("triage_level"),
                    args.get("recommended_workers"),
                )
                return args

            logger.info("Gemma 4 returned no triage tool call — heuristic fallback")
            return None

        except Exception as exc:
            logger.warning("Ollama triage failed (%s) — heuristic fallback", exc)
            return None

    def _heuristic_triage(
        self, query: str, image: Optional[Image.Image]
    ) -> dict:
        """
        Keyword-based routing fallback producing the same schema as _ollama_triage.

        Covers the four cases that arise in tests and offline deployments:
          - Image present      → dermatology, analyze_skin_lesion
          - Derm keyword + Q   → dermatology, retrieve_dermatology_knowledge
          - Derm keyword       → dermatology, analyze_skin_lesion
          - No derm keyword    → general, no workers
        """
        q = query.lower()
        matched = [kw for kw in _DERM_KEYWORDS if kw in q]
        is_derm = bool(matched) or image is not None
        is_urgent = any(kw in q for kw in _URGENT_KEYWORDS)
        is_question = self._is_knowledge_query(query)

        # Build worker list
        workers: list[str] = []
        if image is not None:
            workers.append("analyze_skin_lesion")
        elif is_derm and not is_question:
            workers.append("analyze_skin_lesion")
        if is_derm and is_question:
            workers.append("retrieve_dermatology_knowledge")

        # Urgency is meaningful only for personal symptom queries, not educational
        # questions — e.g. "What is melanoma?" should not be triaged as urgent.
        symptom_urgent = is_urgent and not is_question
        triage = (
            "urgent" if symptom_urgent
            else "moderate" if is_derm
            else "low"
        )

        # Build human-readable reasoning for the UI routing panel.
        reasoning_parts: list[str] = []
        if image is not None:
            reasoning_parts.append("Skin image attached — routed to visual analysis.")
        if matched:
            kw_sample = ", ".join(matched[:3])
            reasoning_parts.append(f"Dermatology terms detected: {kw_sample}.")
        if is_question and is_derm:
            reasoning_parts.append(
                "Query is informational — routed to knowledge retrieval."
            )
        if symptom_urgent:
            reasoning_parts.append(
                "Urgency indicator present in symptom description — triage level elevated."
            )
        if not reasoning_parts:
            reasoning_parts.append(
                "No dermatology indicators detected — general conversational response."
            )

        return {
            "domain": "dermatology" if is_derm else "general",
            "triage_level": triage,
            "extracted_symptoms": matched[:5],
            "recommended_workers": workers,
            "reasoning": " ".join(reasoning_parts),
            "routing_source": "heuristic",
        }

    # ── Legacy single-call routing (preserved for backward-compat tests) ──────

    def _keyword_routing(self, query: str) -> Optional[dict]:
        """
        Lightweight keyword check returning a single tool-call dict or None.
        Called directly by existing tests; not used in the main routing path.
        """
        if any(kw in query.lower() for kw in _DERM_KEYWORDS):
            logger.info("Keyword routing → analyze_skin_lesion")
            return {"name": "analyze_skin_lesion", "arguments": {"symptoms": query}}
        return None

    # ── Phase 3: Synthesis ────────────────────────────────────────────────────

    def _synthesise_response(
        self,
        query: str,
        routing_decision: dict,
        worker_result: Optional[dict],
        rag_result: Optional[dict] = None,
    ) -> str:
        """
        Produce a plain-language response from worker results and triage context.

        - No workers → general conversational response.
        - RAG only   → return RAG answer (already safety-enforced).
        - Analysis (± RAG) → Gemma 4 synthesis using structured template.
        """
        if worker_result is None and rag_result is None:
            return self._general_response(query)

        if worker_result is None and rag_result is not None:
            return rag_result.get("answer", "")

        # Build RAG context section for the synthesis prompt.
        #
        # Priority 1: an explicit RAGWorker answer (Gemma 4-synthesised knowledge)
        #   → pass the full answer so the synthesis LLM can quote or paraphrase it.
        # Priority 2: auto-retrieval via get_context_for_synthesis() (TF-IDF only,
        #   no extra LLM call — safe on CPU-only hardware).  Use up to 500 chars so
        #   Gemma 4 has enough context to ground its response without token bloat.
        rag_section = ""
        if rag_result and rag_result.get("answer"):
            rag_section = (
                "\nRELEVANT KNOWLEDGE CONTEXT (RAG-synthesised):\n"
                + rag_result["answer"]
                + "\n"
            )
        elif worker_result:
            brief_ctx = self.rag_worker.get_context_for_synthesis(query, top_k=2)
            if brief_ctx:
                rag_section = (
                    "\nRELEVANT KNOWLEDGE CONTEXT (auto-retrieved from guidelines KB):\n"
                    + brief_ctx[:500]
                    + "\n"
                )

        prompt = _SYNTHESIS_TEMPLATE.format(
            domain=routing_decision.get("domain", "unknown"),
            triage_level=routing_decision.get("triage_level", "unknown"),
            symptoms=", ".join(routing_decision.get("extracted_symptoms", [])) or "none extracted",
            reasoning=routing_decision.get("reasoning", ""),
            classification=worker_result.get("classification", "unknown"),
            confidence=worker_result.get("confidence", "unknown"),
            risk_level=worker_result.get("risk_level", "unknown"),
            abcde_notes=worker_result.get("abcde_notes", "N/A"),
            plain_explanation=worker_result.get("plain_explanation", "N/A"),
            recommended_action=worker_result.get("recommended_action", "N/A"),
            rag_section=rag_section,
            query=query,
        )

        if self.use_ollama and self._client:
            try:
                resp = self._client.chat(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": _SYNTHESIS_SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    options={"temperature": 0.3},
                )
                return resp.message.content
            except Exception as exc:
                logger.warning("Ollama synthesis failed (%s) — template", exc)

        return self._template_synthesis(worker_result, routing_decision)

    def _general_response(self, query: str) -> str:
        if self.use_ollama and self._client:
            try:
                messages = [{"role": "system", "content": _SYNTHESIS_SYSTEM}]
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

    def _template_synthesis(self, result: dict, routing_decision: dict) -> str:
        cls = result.get("classification", "unclassified").replace("_", " ").title()
        conf = result.get("confidence", "unknown")
        risk = result.get("risk_level", "unknown")
        explanation = result.get("plain_explanation", "No explanation available.")
        action = result.get("recommended_action", "Please consult a dermatologist.")
        triage = routing_decision.get("triage_level", "unknown")

        urgency_note = (
            "⚠️ Triage level is URGENT — please seek professional evaluation today "
            "or within the next few days.\n\n"
            if triage == "urgent"
            else ""
        )

        parts = [
            urgency_note,
            f"The image analysis suggests: **{cls}** "
            f"(confidence: {conf}, risk level: {risk}).",
            "",
            explanation,
            "",
            f"Recommended next step: {action}",
        ]
        return "\n".join(parts)

    # ── Utilities ─────────────────────────────────────────────────────────────

    @staticmethod
    def _is_knowledge_query(text: str) -> bool:
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
