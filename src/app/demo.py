"""
Gradio demo interface for G4G RuralClinic AI — Offline Dermatology Assistant.

Two-tab layout:
  Tab 1 — Image Analysis: upload a skin image + describe symptoms, get a
           structured analysis with classification, ABCDE notes, and
           recommended action.  Full function-call trace shown for transparency.
  Tab 2 — Knowledge Chat: ask plain-language dermatology questions answered
           by the RAG worker using the offline knowledge base.

Features:
  - Offline toggle: restricts external calls after initial model pull.
  - Persistent safety disclaimer on every response.
  - Lazy model loading: Ollama/HF backend initialised on first request.

Usage:
    python -m src.app.demo

Opens at http://localhost:7860.

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

# Lazy singletons — loaded on first request, not at import time.
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
        from src.workers.rag_worker import RAGWorker
        _rag_worker = RAGWorker(use_ollama=True)
    return _rag_worker


# ─── Tab 1: Image Analysis ────────────────────────────────────────────────────

def run_analysis(
    image: Optional[Image.Image],
    symptoms: str,
    offline_mode: bool,
) -> tuple[str, str, str]:
    """
    Gradio callback for the Analyse button.

    Returns:
        (result_markdown, trace_markdown, disclaimer_markdown)
    """
    os.environ["OFFLINE_MODE"] = "true" if offline_mode else "false"

    if image is None and not symptoms.strip():
        return (
            "*Please upload a skin image, describe your symptoms, or both.*",
            "*No analysis run yet.*",
            SAFETY_DISCLAIMER,
        )

    try:
        orch = _get_orchestrator()
        effective_query = symptoms.strip() or "Please analyse this skin image."
        result = orch.route_and_run(query=effective_query, image=image)

        # ── Format result panel ───────────────────────────────────────────
        if result.get("worker_result"):
            detail = format_for_display(result["worker_result"])
            synthesis = result.get("response", "").strip()
            if synthesis and len(synthesis) > 30:
                display = (
                    "### AI Summary\n" + synthesis
                    + "\n\n---\n\n### Detailed Analysis\n" + detail
                )
            else:
                display = detail
        elif result.get("rag_result"):
            # Image was None but RAG answered a knowledge question
            display = result.get("response", "*No response generated.*")
        else:
            display = result.get("response", "*No response generated.*")

        if result.get("error"):
            display = f"**Error:** {result['error']}\n\n{display}"

        # ── Format function-call trace panel ─────────────────────────────
        trace_md = _format_trace(result.get("function_calls", []))

        return display, trace_md, SAFETY_DISCLAIMER

    except Exception as exc:
        logger.exception("Analysis error: %s", exc)
        return (
            f"**An error occurred:** {exc}\n\n"
            f"Ensure Ollama is running (`ollama serve`) and the model is available "
            f"(`ollama pull {MODEL_NAME}`).",
            "*Error — no function calls completed.*",
            SAFETY_DISCLAIMER,
        )


def clear_analysis() -> tuple[str, str, str]:
    global _orchestrator
    if _orchestrator is not None:
        _orchestrator.clear_history()
    return (
        "*History cleared. Upload an image or describe symptoms to begin.*",
        "*No analysis run yet.*",
        SAFETY_DISCLAIMER,
    )


# ─── Tab 2: Knowledge Chat ────────────────────────────────────────────────────

def ask_knowledge(
    question: str,
    offline_mode: bool,
) -> tuple[str, str, str]:
    """
    Gradio callback for the Ask button in the Knowledge Chat tab.

    Returns:
        (answer_markdown, sources_markdown, disclaimer_markdown)
    """
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

        sources = result.get("sources", [])
        if sources:
            src_lines = ["### Knowledge Sources Retrieved\n"]
            for i, s in enumerate(sources, 1):
                cond = (s.get("condition") or "general").replace("_", " ").title()
                title = s.get("title", "Unknown")
                score = s.get("score", 0)
                src_lines.append(f"**{i}. {title}** *(condition: {cond}, relevance: {score:.2f})*")
            sources_md = "\n".join(src_lines)
        else:
            sources_md = "*No relevant knowledge chunks found.*"

        return answer, sources_md, SAFETY_DISCLAIMER

    except Exception as exc:
        logger.exception("Knowledge chat error: %s", exc)
        return (
            f"**An error occurred:** {exc}",
            "*Error retrieving sources.*",
            SAFETY_DISCLAIMER,
        )


def clear_knowledge() -> tuple[str, str, str]:
    return (
        "*Enter a question about a skin condition to begin.*",
        "*No sources retrieved.*",
        SAFETY_DISCLAIMER,
    )


# ─── Shared helpers ───────────────────────────────────────────────────────────

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


# ─── UI construction ──────────────────────────────────────────────────────────

def build_demo():
    """Build and return the Gradio Blocks interface."""
    import gradio as gr

    _CSS = """
    .disclaimer { background:#fff8e1; border:1px solid #f9a825;
                  border-radius:6px; padding:10px; font-size:0.88em; }
    .result-panel { line-height:1.65; }
    .offline-badge { font-size:0.82em; color:#555; }
    """

    with gr.Blocks(
        title="G4G RuralClinic AI — Dermatology Assistant",
        theme=gr.themes.Soft(),
        css=_CSS,
    ) as demo:

        gr.Markdown(
            "# G4G RuralClinic AI — Offline Dermatology Assistant\n"
            "**Powered by Gemma 4 · Fully Offline · Educational Use Only**"
        )

        # ── Shared offline toggle ─────────────────────────────────────────
        with gr.Row():
            offline_toggle = gr.Checkbox(
                label="Offline Mode (no internet after initial model pull)",
                value=OFFLINE_MODE,
                info="All inference runs locally — no data leaves your machine.",
                scale=2,
            )
            gr.Markdown(
                f"*Model: `{MODEL_NAME}` · "
                "Run `ollama serve` then `ollama pull gemma4:e4b` before first use.*",
                elem_classes=["offline-badge"],
                scale=3,
            )

        # ── Tabs ──────────────────────────────────────────────────────────
        with gr.Tabs():

            # ── Tab 1: Image Analysis ─────────────────────────────────────
            with gr.TabItem("Image Analysis"):
                gr.Markdown(
                    "Upload a skin lesion image and/or describe symptoms. "
                    "Gemma 4 will analyse the image using multimodal function calling "
                    "and provide an informational assessment — **not a diagnosis**."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        image_input = gr.Image(
                            type="pil",
                            label="Skin Lesion Image",
                            height=280,
                        )
                        symptoms_input = gr.Textbox(
                            label="Symptom Description (optional)",
                            placeholder=(
                                "e.g. 'Dark mole on upper arm, growing and changing "
                                "colour over 3 months. Sometimes itchy.'"
                            ),
                            lines=4,
                        )
                        with gr.Row():
                            analyse_btn = gr.Button("Analyse", variant="primary", scale=3)
                            clear_btn = gr.Button("Clear", scale=1)

                    with gr.Column(scale=1):
                        result_output = gr.Markdown(
                            value="*Upload an image or enter symptoms to begin.*",
                            label="Analysis Result",
                            elem_classes=["result-panel"],
                        )
                        with gr.Accordion("Function Call Trace", open=False):
                            trace_output = gr.Markdown(value="*No analysis run yet.*")

                disclaimer_analysis = gr.Markdown(
                    value=SAFETY_DISCLAIMER,
                    elem_classes=["disclaimer"],
                )

                analyse_btn.click(
                    fn=run_analysis,
                    inputs=[image_input, symptoms_input, offline_toggle],
                    outputs=[result_output, trace_output, disclaimer_analysis],
                )
                clear_btn.click(
                    fn=clear_analysis,
                    inputs=[],
                    outputs=[result_output, trace_output, disclaimer_analysis],
                )

            # ── Tab 2: Knowledge Chat ─────────────────────────────────────
            with gr.TabItem("Knowledge Chat"):
                gr.Markdown(
                    "Ask any dermatology question. Answers are grounded in an **offline "
                    "knowledge base** — no image required. "
                    "Try: *'What is the ABCDE rule?'*, *'How do I check a mole?'*, "
                    "*'What causes psoriasis?'*"
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder=(
                                "e.g. 'What is melanoma?', 'When should I see a dermatologist?', "
                                "'How can I protect my skin from UV damage?'"
                            ),
                            lines=3,
                        )
                        with gr.Row():
                            ask_btn = gr.Button("Ask", variant="primary", scale=3)
                            clear_kb_btn = gr.Button("Clear", scale=1)

                    with gr.Column(scale=1):
                        answer_output = gr.Markdown(
                            value="*Enter a question to begin.*",
                            label="Answer",
                            elem_classes=["result-panel"],
                        )
                        with gr.Accordion("Retrieved Knowledge Sources", open=False):
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
            "**How it works**\n\n"
            "1. Your data is processed entirely on your local machine.\n"
            "2. **Image Analysis** uses Gemma 4 function calling to decide whether to "
            "   invoke the dermatology specialist worker (visual + ABCDE assessment).\n"
            "3. **Knowledge Chat** retrieves relevant excerpts from an offline "
            "   TF-IDF-indexed knowledge base and uses Gemma 4 to synthesise a grounded answer.\n"
            "4. All outputs carry mandatory safety guardrails and a medical disclaimer.\n\n"
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
