"""
Core orchestrator for G4G RuralClinic AI.

Responsibilities:
  1. Load Gemma 4 (via Ollama or HuggingFace pipeline).
  2. Use Gemma 4's native function-calling API to decide which specialist
     worker to invoke for a given query + image.
  3. Dispatch to the chosen worker and collect structured results.
  4. Synthesise a final plain-language response with safety guardrails.
  5. Maintain conversation history and produce a traceable function-call log.

Function-calling flow:
  User query + image
      → Gemma 4 decides: call `analyze_skin_lesion`?
          → DermatologyWorker.analyze(image, symptoms)
              → Structured JSON result
                  → Gemma 4 synthesises plain-language response
                      → Safety enforcement + disclaimer
                          → Final output dict

Project: G4G RuralClinic AI — offline frontier AI for accessible
dermatology screening in rural and underserved communities.
"""

import logging
from datetime import datetime
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

logger = logging.getLogger(__name__)

# ─── System prompt ────────────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a compassionate AI health assistant for rural and underserved communities.
Your role is to help users understand skin conditions using image analysis and symptom descriptions.

ABSOLUTE RULES — never break these:
1. You are NOT a doctor. Never make a definitive medical diagnosis.
2. Always recommend consulting a qualified dermatologist or healthcare professional.
3. When the user provides a skin image OR describes a skin symptom, call the `analyze_skin_lesion` tool.
4. Explain findings in plain language — avoid jargon.
5. Be empathetic, supportive, and honest about uncertainty.
6. If you are unsure, say so clearly."""

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

Write a compassionate, plain-language summary for the user (under 180 words).
Emphasise that this is NOT a diagnosis and they MUST consult a dermatologist.
Start with what was found, then what it might mean, then what to do next."""


class Orchestrator:
    """
    Central coordinator for the RuralClinic AI system.

    Manages Gemma 4 inference, function-call routing, worker dispatch,
    response synthesis, and conversation history in a single stateful object.

    Typical usage::

        orch = Orchestrator()
        result = orch.route_and_run("Is this mole dangerous?", image=pil_image)
        print(result["response"])
    """

    def __init__(self, use_ollama: bool = True):
        """
        Initialise the orchestrator and load the Gemma 4 backend.

        Args:
            use_ollama: Prefer Ollama if True; fall back to HF pipeline.
        """
        self.use_ollama = use_ollama
        self.history: list[dict] = []
        self.function_call_log: list[dict] = []

        self.dermatology_worker = DermatologyWorker(use_ollama=use_ollama)

        self._client = None        # Ollama client
        self._model = None         # HF model
        self._processor = None     # HF processor

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
        """Create Ollama client and verify the model is available locally."""
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
        """Load Gemma 4 via HuggingFace transformers (Ollama is strongly preferred)."""
        try:
            import torch
            from transformers import AutoProcessor

            # AutoModelForImageTextToText is correct for Gemma 4 >= transformers 4.50.
            # Fall back to AutoModelForCausalLM on older installs with a loud warning.
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
        Primary entry point: route a user query + optional image through
        Gemma 4 function calling, execute the appropriate worker, and return
        a synthesised plain-language response.

        Args:
            query: User's text question or symptom description.
            image: Optional PIL Image of a skin lesion.

        Returns:
            dict with keys:
              response        (str)  — Final synthesised answer for the user.
              worker_result   (dict) — Raw structured output from the worker, or None.
              function_calls  (list) — Audit trail of all tool calls made.
              disclaimer      (str)  — Medical safety disclaimer.
              timestamp       (str)  — UTC ISO-8601 timestamp.
              error           (str)  — Present only if an exception occurred.
        """
        timestamp = datetime.utcnow().isoformat()
        logger.info(
            "route_and_run | has_image=%s | query=%.80s", image is not None, query
        )
        self.function_call_log.clear()

        try:
            # Step 1: Ask Gemma 4 whether a tool call is needed
            tool_call = self._decide_tool_call(query, image)

            # Step 2: Execute the worker if a tool was selected
            worker_result = None
            if tool_call:
                worker_result = self._dispatch_tool(tool_call, image)
                self._log_function_call(
                    tool_call["name"],
                    tool_call.get("arguments", {}),
                    worker_result,
                )

            # Step 3: Synthesise the final response
            response = self._synthesise_response(query, worker_result)
            response = enforce_safety(response)

            self._update_history(query, response)

            return {
                "response": response,
                "worker_result": worker_result,
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
                "function_calls": list(self.function_call_log),
                "disclaimer": SAFETY_DISCLAIMER,
                "timestamp": timestamp,
                "error": str(exc),
            }

    # ── Tool-call decision ────────────────────────────────────────────────────

    def _decide_tool_call(
        self, query: str, image: Optional[Image.Image]
    ) -> Optional[dict]:
        """
        Determine whether a worker tool should be called.

        Priority order:
          1. Image present → always route to dermatology worker.
          2. Ollama available → use Gemma 4 native function calling.
          3. Fallback → keyword-based routing.

        Returns:
            dict with 'name' and 'arguments' keys, or None for general queries.
        """
        if image is not None:
            logger.info("Image present — auto-routing to analyze_skin_lesion")
            return {"name": "analyze_skin_lesion", "arguments": {"symptoms": query}}

        if self.use_ollama and self._client:
            return self._ollama_tool_decision(query)

        return self._keyword_routing(query)

    def _ollama_tool_decision(self, query: str) -> Optional[dict]:
        """
        Use Ollama's native tool-calling API to let Gemma 4 decide
        whether to invoke a specialist worker.

        Gemma 4 returns a `tool_calls` block when it selects a tool.
        Returns None if the model decides no tool is needed.
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
            if (
                hasattr(response, "message")
                and response.message.tool_calls
            ):
                tc = response.message.tool_calls[0]
                logger.info("Gemma 4 selected tool: %s", tc.function.name)
                return {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or {},
                }
            logger.info("Gemma 4 selected no tool — general response")
            return None

        except Exception as exc:
            logger.warning("Ollama tool decision failed (%s) — using keyword routing", exc)
            return self._keyword_routing(query)

    def _keyword_routing(self, query: str) -> Optional[dict]:
        """
        Lightweight fallback routing based on dermatology keyword matching.
        Used when the model backend is unavailable or returns no tool call.
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

    # ── Worker dispatch ───────────────────────────────────────────────────────

    def _dispatch_tool(
        self, tool_call: dict, image: Optional[Image.Image]
    ) -> dict:
        """
        Route a parsed tool-call dict to the appropriate specialist worker.

        Args:
            tool_call: {'name': str, 'arguments': dict}
            image: PIL Image if attached to the request.

        Returns:
            Worker output dict.

        Raises:
            ValueError: If the tool name is not recognised.
        """
        name = tool_call.get("name")
        args = tool_call.get("arguments", {})

        if name == "analyze_skin_lesion":
            symptoms = args.get("symptoms", "")
            return self.dermatology_worker.analyze(image=image, symptoms=symptoms)

        raise ValueError(f"Unknown tool requested by model: '{name}'")

    # ── Response synthesis ────────────────────────────────────────────────────

    def _synthesise_response(
        self, query: str, worker_result: Optional[dict]
    ) -> str:
        """
        Use Gemma 4 to write a compassionate plain-language summary from
        the worker's structured output, or generate a general response when
        no worker was invoked.
        """
        if worker_result is None:
            return self._general_response(query)

        prompt = _SYNTHESIS_PROMPT_TEMPLATE.format(
            query=query,
            classification=worker_result.get("classification", "unknown"),
            confidence=worker_result.get("confidence", "unknown"),
            risk_level=worker_result.get("risk_level", "unknown"),
            visual_description=worker_result.get("visual_description", "N/A"),
            abcde_notes=worker_result.get("abcde_notes", "N/A"),
            plain_explanation=worker_result.get("plain_explanation", "N/A"),
            recommended_action=worker_result.get("recommended_action", "N/A"),
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
                logger.warning("Ollama synthesis failed (%s) — using template", exc)

        # Fallback: template-based synthesis without the LLM
        return self._template_synthesis(worker_result)

    def _general_response(self, query: str) -> str:
        """Generate a conversational response for queries that don't need a worker."""
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
        """Build a simple plain-text summary without calling the LLM."""
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

    def _log_function_call(self, name: str, arguments: dict, result: dict):
        """Record a tool call in the audit log for UI display and traceability."""
        entry = {
            "tool": name,
            "arguments": arguments,
            "result_keys": list(result.keys()) if result else [],
            "timestamp": datetime.utcnow().isoformat(),
        }
        self.function_call_log.append(entry)
        logger.info(
            "FUNCTION_CALL | tool=%s | arg_keys=%s | result_keys=%s",
            name,
            list(arguments.keys()),
            entry["result_keys"],
        )

    def _update_history(self, query: str, response: str):
        """Append this turn to conversation history, trimming if over limit."""
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})
        max_entries = MAX_HISTORY_LENGTH * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    def clear_history(self):
        """Reset conversation history (e.g. when the user clicks 'Clear')."""
        self.history.clear()
        logger.info("Conversation history cleared")
