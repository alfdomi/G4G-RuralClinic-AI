"""
Gradio demo interface for G4G RuralClinic AI — Offline Dermatology Assistant.

Two-tab layout:

  Tab 1 — Image Analysis
  ──────────────────────
  Upload a skin image and/or describe symptoms. The orchestrator runs the
  two-phase agentic workflow:
    Phase 1: Gemma 4 function calling → structured routing_decision JSON.
    Phase 2: DermatologyWorker (ABCDE analysis) + optional RAGWorker.
  UI shows:
    • Routing Decision panel: domain badge, triage-level colour, extracted
      symptom chips, recommended workers list, routing reasoning.
    • AI Summary: Gemma 4 synthesised plain-language response.
    • Detailed Analysis: DermatologyWorker structured output.
    • Knowledge Context accordion: RAGWorker sources if retrieved.
    • Function Call Trace accordion: full audit log.
    • Safety disclaimer (always visible, prominent).

  Tab 2 — Knowledge Chat
  ──────────────────────
  Ask any dermatology question. RAGWorker v2 retrieves from the
  comprehensive offline guidelines KB (45 chunks including skin-of-color,
  Fitzpatrick types, rural/tropical conditions, WHO criteria) and Gemma 4
  synthesises a grounded answer. Sources panel shows what was retrieved.

Shared: Offline Mode toggle, model info strip.

Usage:
    python -m src.app.demo   →  http://localhost:7860

Project: G4G RuralClinic AI — offline frontier AI for accessible
dermatology screening in rural and underserved communities.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

from src.config import MODEL_NAME, OFFLINE_MODE, SAFETY_DISCLAIMER
from src.utils.safety import format_for_display

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Lazy singletons — heavy model loading deferred to first request.
_orchestrator = None
_rag_worker = None


def _get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        logger.info("Initialising Orchestrator (first request)…")
        from src.orchestrator.core import Orchestrator
        _orchestrator = Orchestrator(use_ollama=True)
    return _orchestrator


def _get_rag_worker():
    global _rag_worker
    if _rag_worker is None:
        logger.info("Initialising RAGWorker (first Knowledge Chat request)…")
        from src.workers.rag import RAGWorker
        _rag_worker = RAGWorker(use_ollama=True)
    return _rag_worker


# ─── Triage-level formatting ──────────────────────────────────────────────────
_TRIAGE_ICONS = {
    "urgent":   "🔴 **URGENT**",
    "moderate": "🟡 **Moderate**",
    "low":      "🟢 Low",
    "unclear":  "⚪ Unclear",
}
_DOMAIN_ICONS = {
    "dermatology": "🔬 Dermatology",
    "general":     "💬 General",
    "unclear":     "❓ Unclear",
}


def _format_routing_panel(routing_decision: dict) -> str:
    """Render the routing_decision dict as a readable Markdown block."""
    if not routing_decision:
        return "*No routing decision available.*"

    domain = routing_decision.get("domain", "unclear")
    triage = routing_decision.get("triage_level", "unclear")
    symptoms = routing_decision.get("extracted_symptoms", [])
    workers = routing_decision.get("recommended_workers", [])
    reasoning = routing_decision.get("reasoning", "")
    source = routing_decision.get("routing_source", "unknown")

    domain_label = _DOMAIN_ICONS.get(domain, f"❓ {domain}")
    triage_label = _TRIAGE_ICONS.get(triage, f"⚪ {triage}")

    symptom_chips = (
        " · ".join(f"`{s}`" for s in symptoms) if symptoms else "*none extracted*"
    )
    worker_list = (
        " + ".join(f"`{w}`" for w in workers) if workers else "*none — general response*"
    )

    source_note = (
        "*(decision by Gemma 4 function call)*"
        if source == "gemma4_function_call"
        else "*(heuristic fallback)*"
    )

    lines = [
        f"| Field | Value |",
        f"|---|---|",
        f"| **Domain** | {domain_label} |",
        f"| **Triage Level** | {triage_label} |",
        f"| **Extracted Symptoms** | {symptom_chips} |",
        f"| **Workers Invoked** | {worker_list} |",
        f"",
        f"**Routing Reasoning** {source_note}",
        f"> {reasoning}",
    ]
    return "\n".join(lines)


def _format_rag_sources(rag_result: Optional[dict]) -> str:
    """Render RAGWorker source metadata as Markdown."""
    if not rag_result:
        return "*No knowledge context retrieved.*"
    sources = rag_result.get("sources", [])
    if not sources:
        return "*No relevant knowledge chunks found.*"
    lines = ["### Retrieved Knowledge Sources\n"]
    for i, s in enumerate(sources, 1):
        cond = (s.get("condition") or "general").replace("_", " ").title()
        title = s.get("title", "Unknown")
        score = s.get("score", 0)
        domain = s.get("domain", "")
        lines.append(
            f"**{i}. {title}**  \n"
            f"*Condition: {cond} · Domain: {domain} · Relevance: {score:.2f}*"
        )
    return "\n\n".join(lines)


def _format_trace(calls: list[dict]) -> str:
    if not calls:
        return "*No tool calls made — query handled as a general question.*"
    parts = ["### Function Calls Traced\n"]
    for i, call in enumerate(calls, 1):
        args_str = json.dumps(call.get("arguments", {}), indent=2)
        parts.append(
            f"**{i}. Tool:** `{call['tool']}`  \n"
            f"**Arguments:**\n```json\n{args_str}\n```  \n"
            f"**Result keys:** `{call.get('result_keys', [])}`  \n"
            f"**Timestamp:** {call.get('timestamp', 'N/A')}\n"
        )
    return "\n".join(parts)


# ─── Tab 1: Image Analysis ────────────────────────────────────────────────────

def run_analysis(
    image: Optional[Image.Image],
    symptoms: str,
    offline_mode: bool,
) -> tuple[str, str, str, str, str, str]:
    """
    Gradio callback for the Analyse button.

    Returns:
        (routing_md, result_md, rag_sources_md, trace_md, disclaimer_md)
    """
    os.environ["OFFLINE_MODE"] = "true" if offline_mode else "false"

    if image is None and not symptoms.strip():
        empty = "*Please upload a skin image, describe your symptoms, or both.*"
        return empty, empty, "*No knowledge context.*", "*No trace.*", SAFETY_DISCLAIMER

    try:
        orch = _get_orchestrator()
        effective_query = symptoms.strip() or "Please analyse this skin image."
        result = orch.route_and_run(query=effective_query, image=image)

        # ── Routing decision panel ────────────────────────────────────────
        routing_md = _format_routing_panel(result.get("routing_decision", {}))

        # ── Result panel ──────────────────────────────────────────────────
        if result.get("worker_result"):
            detail = format_for_display(result["worker_result"])
            synthesis = result.get("response", "").strip()
            if synthesis and len(synthesis) > 30:
                result_md = (
                    "### AI Summary\n" + synthesis
                    + "\n\n---\n\n### Detailed Analysis\n" + detail
                )
            else:
                result_md = detail
        elif result.get("rag_result"):
            result_md = result.get("response", "*No response generated.*")
        else:
            result_md = result.get("response", "*No response generated.*")

        if result.get("error"):
            result_md = f"**Error:** {result['error']}\n\n{result_md}"

        # ── Knowledge context panel ───────────────────────────────────────
        rag_sources_md = _format_rag_sources(result.get("rag_result"))

        # ── Trace panel ───────────────────────────────────────────────────
        trace_md = _format_trace(result.get("function_calls", []))

        return routing_md, result_md, rag_sources_md, trace_md, SAFETY_DISCLAIMER

    except Exception as exc:
        logger.exception("Analysis error: %s", exc)
        err = (
            f"**An error occurred:** {exc}\n\n"
            f"Ensure Ollama is running (`ollama serve`) and the model is available "
            f"(`ollama pull {MODEL_NAME}`)."
        )
        return err, err, "*Error.*", "*Error.*", SAFETY_DISCLAIMER


def clear_analysis() -> tuple[str, str, str, str, str]:
    global _orchestrator
    if _orchestrator is not None:
        _orchestrator.clear_history()
    empty = "*History cleared. Upload an image or describe symptoms to begin.*"
    return empty, empty, "*No knowledge context.*", "*No analysis run yet.*", SAFETY_DISCLAIMER


# ─── Tab 2: Knowledge Chat ────────────────────────────────────────────────────

def ask_knowledge(
    question: str,
    offline_mode: bool,
) -> tuple[str, str, str]:
    os.environ["OFFLINE_MODE"] = "true" if offline_mode else "false"

    question = question.strip()
    if not question:
        return (
            "*Please enter a question about a skin condition.*",
            "*No sources retrieved.*",
            SAFETY_DISCLAIMER,
        )

    try:
        rag = _get_rag_worker()
        result = rag.answer(question)
        answer = result.get("answer", "*No answer generated.*")
        sources_md = _format_rag_sources(result)
        return answer, sources_md, SAFETY_DISCLAIMER

    except Exception as exc:
        logger.exception("Knowledge chat error: %s", exc)
        return f"**An error occurred:** {exc}", "*Error.*", SAFETY_DISCLAIMER


def clear_knowledge() -> tuple[str, str, str]:
    return (
        "*Enter a question about a skin condition to begin.*",
        "*No sources retrieved.*",
        SAFETY_DISCLAIMER,
    )


# ─── UI construction ──────────────────────────────────────────────────────────

def build_demo():
    """Build and return the Gradio Blocks interface."""
    import gradio as gr

    _CSS = """
    .disclaimer {
        background: #fff8e1; border: 1px solid #f9a825;
        border-radius: 6px; padding: 10px; font-size: 0.88em;
    }
    .result-panel { line-height: 1.7; }
    .routing-panel {
        background: #f0f4ff; border: 1px solid #c5cae9;
        border-radius: 6px; padding: 10px;
    }
    .model-info { font-size: 0.82em; color: #555; }
    """

    with gr.Blocks(
        title="G4G RuralClinic AI — Dermatology Assistant",
        theme=gr.themes.Soft(),
        css=_CSS,
    ) as demo:

        gr.Markdown(
            "# G4G RuralClinic AI — Offline Dermatology Assistant\n"
            "**Powered by Gemma 4 Native Function Calling · Fully Offline · Educational Use Only**"
        )

        # ── Shared header: offline toggle + model info ─────────────────────
        with gr.Row():
            offline_toggle = gr.Checkbox(
                label="Offline Mode",
                value=OFFLINE_MODE,
                info="All inference runs locally — no data leaves your device.",
                scale=1,
            )
            gr.Markdown(
                f"Model: `{MODEL_NAME}` · "
                "Run `ollama serve` + `ollama pull gemma4:e4b` before first use.  \n"
                "Phase 1: `triage_and_route` (structured JSON routing) → "
                "Phase 2: specialist workers → Phase 3: Gemma 4 synthesis.",
                elem_classes=["model-info"],
                scale=4,
            )

        with gr.Tabs():

            # ── Tab 1: Image Analysis ──────────────────────────────────────
            with gr.TabItem("🔬 Image Analysis"):
                gr.Markdown(
                    "Upload a skin lesion image and/or describe symptoms. "
                    "Gemma 4 runs a **two-phase agentic workflow**: "
                    "Phase 1 produces a structured routing decision via native function calling; "
                    "Phase 2 dispatches to the ABCDE analysis worker and/or knowledge RAG worker."
                )
                with gr.Row():
                    # Left: inputs
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Skin Lesion Image",
                            height=280,
                        )
                        symptoms_input = gr.Textbox(
                            label="Symptom Description (optional)",
                            placeholder=(
                                "e.g. 'Dark mole on upper arm, asymmetric, multiple colours, "
                                "growing over 3 months. Sometimes itchy and bleeds.'"
                            ),
                            lines=4,
                        )
                        with gr.Row():
                            analyse_btn = gr.Button("Analyse", variant="primary", scale=3)
                            clear_btn = gr.Button("Clear", scale=1)

                    # Right: outputs
                    with gr.Column(scale=1):
                        with gr.Accordion(
                            "🗺️ Phase 1 — Routing Decision", open=True
                        ):
                            routing_output = gr.Markdown(
                                value="*Run an analysis to see the routing decision.*",
                                elem_classes=["routing-panel"],
                            )
                        result_output = gr.Markdown(
                            value="*Upload an image or enter symptoms to begin.*",
                            label="Phase 2+3 — Analysis & AI Summary",
                            elem_classes=["result-panel"],
                        )
                        with gr.Accordion("📚 Knowledge Context", open=False):
                            rag_output = gr.Markdown(value="*No knowledge context retrieved.*")
                        with gr.Accordion("🔍 Function Call Trace", open=False):
                            trace_output = gr.Markdown(value="*No analysis run yet.*")

                disclaimer_analysis = gr.Markdown(
                    value=SAFETY_DISCLAIMER,
                    elem_classes=["disclaimer"],
                )

                analyse_btn.click(
                    fn=run_analysis,
                    inputs=[image_input, symptoms_input, offline_toggle],
                    outputs=[
                        routing_output,
                        result_output,
                        rag_output,
                        trace_output,
                        disclaimer_analysis,
                    ],
                )
                clear_btn.click(
                    fn=clear_analysis,
                    inputs=[],
                    outputs=[
                        routing_output,
                        result_output,
                        rag_output,
                        trace_output,
                        disclaimer_analysis,
                    ],
                )

            # ── Tab 2: Knowledge Chat ──────────────────────────────────────
            with gr.TabItem("💬 Knowledge Chat"):
                gr.Markdown(
                    "Ask any dermatology question. Answers are grounded in an "
                    "**offline guidelines knowledge base** (45 chunks covering ABCDE criteria, "
                    "all skin classes, skin-of-color presentations, Fitzpatrick types, "
                    "rural/tropical conditions, and WHO referral criteria). No image needed.  \n"
                    "Try: *'How does melanoma present on dark skin?'*, "
                    "*'What is the Fitzpatrick scale?'*, "
                    "*'How do I treat scabies in a rural clinic?'*"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder=(
                                "e.g. 'What is the ABCDE rule?', "
                                "'How does eczema look on darker skin?', "
                                "'When do I need urgent referral?'"
                            ),
                            lines=3,
                        )
                        with gr.Row():
                            ask_btn = gr.Button("Ask", variant="primary", scale=3)
                            clear_kb_btn = gr.Button("Clear", scale=1)

                    with gr.Column(scale=1):
                        answer_output = gr.Markdown(
                            value="*Enter a question to begin.*",
                            label="Answer (RAG + Gemma 4)",
                            elem_classes=["result-panel"],
                        )
                        with gr.Accordion("📖 Retrieved Knowledge Sources", open=False):
                            sources_output = gr.Markdown(value="*No sources retrieved.*")

                disclaimer_knowledge = gr.Markdown(
                    value=SAFETY_DISCLAIMER,
                    elem_classes=["disclaimer"],
                )

                ask_btn.click(
                    fn=ask_knowledge,
                    inputs=[question_input, offline_toggle],
                    outputs=[answer_output, sources_output, disclaimer_knowledge],
                )
                clear_kb_btn.click(
                    fn=clear_knowledge,
                    inputs=[],
                    outputs=[answer_output, sources_output, disclaimer_knowledge],
                )

        # ── Footer ────────────────────────────────────────────────────────
        gr.Markdown(
            "---\n"
            "**Agentic Workflow**\n\n"
            "1. **Phase 1 — Triage**: Gemma 4 calls `triage_and_route` → structured JSON "
            "   *(domain, triage_level, extracted_symptoms, recommended_workers, reasoning)*.\n"
            "2. **Phase 2 — Dispatch**: `analyze_skin_lesion` (ABCDE multimodal analysis) "
            "   and/or `retrieve_dermatology_knowledge` (offline TF-IDF + Gemma 4 synthesis).\n"
            "3. **Phase 3 — Synthesis**: Gemma 4 produces a compassionate plain-language "
            "   response enriched with triage context and retrieved guidelines.\n"
            "4. **Safety**: All outputs pass through safety enforcement — overconfident language "
            "   is softened and a mandatory medical disclaimer is appended.\n\n"
            "*G4G RuralClinic AI — bringing offline frontier AI to underserved communities.*"
        )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    logger.info("Starting G4G RuralClinic AI demo on http://localhost:7860")
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )


if __name__ == "__main__":
    main()
