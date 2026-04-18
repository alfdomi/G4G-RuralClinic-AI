"""
Gradio demo interface for G4G RuralClinic AI — Offline Dermatology Assistant.

Features:
  - Image upload for skin lesion analysis
  - Free-text symptom description box
  - "Analyze" button that calls orchestrator.route_and_run()
  - Structured result panel with classification, ABCDE notes, and recommendation
  - Function-call trace panel for transparency/auditability
  - Offline toggle checkbox
  - Persistent safety disclaimer

Usage:
    python -m src.app.demo

The Gradio server starts at http://localhost:7860 by default.
No internet connection is required after the initial model pull.

Project: G4G RuralClinic AI — offline frontier AI for accessible
dermatology screening in rural and underserved communities.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

# Ensure the project root is importable regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image

from src.config import MODEL_NAME, OFFLINE_MODE, SAFETY_DISCLAIMER
from src.utils.safety import format_for_display

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Lazy singleton — model loads on first Analyze click, not at import time.
_orchestrator = None


def _get_orchestrator():
    """Return the shared Orchestrator instance, initialising it if needed."""
    global _orchestrator
    if _orchestrator is None:
        logger.info("Initialising Orchestrator (first request)…")
        from src.orchestrator.core import Orchestrator
        _orchestrator = Orchestrator(use_ollama=True)
    return _orchestrator


# ─── Gradio callbacks ─────────────────────────────────────────────────────────

def run_analysis(
    image: Optional[Image.Image],
    symptoms: str,
    offline_mode: bool,
) -> tuple[str, str, str]:
    """
    Main Gradio callback triggered by the Analyze button.

    Args:
        image: PIL Image uploaded by the user, or None.
        symptoms: Free-text symptom description.
        offline_mode: Whether to restrict internet access.

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
                display = f"### AI Summary\n{synthesis}\n\n---\n\n### Detailed Analysis\n{detail}"
            else:
                display = detail
        else:
            display = result.get("response", "*No response generated.*")

        if result.get("error"):
            display = f"**Error:** {result['error']}\n\n{display}"

        # ── Format trace panel ────────────────────────────────────────────
        calls = result.get("function_calls", [])
        if calls:
            trace_parts = ["### Function Calls Traced\n"]
            for i, call in enumerate(calls, 1):
                args_str = json.dumps(call.get("arguments", {}), indent=2)
                trace_parts.append(
                    f"**{i}. Tool:** `{call['tool']}`  \n"
                    f"**Arguments:**\n```json\n{args_str}\n```  \n"
                    f"**Result keys:** `{call.get('result_keys', [])}`  \n"
                    f"**Timestamp:** {call.get('timestamp', 'N/A')}\n"
                )
            trace_md = "\n".join(trace_parts)
        else:
            trace_md = "*No tool calls made — query handled as a general question.*"

        return display, trace_md, SAFETY_DISCLAIMER

    except Exception as exc:
        logger.exception("Demo error: %s", exc)
        error_display = (
            f"**An error occurred:** {exc}\n\n"
            f"Please ensure Ollama is running (`ollama serve`) and "
            f"the model is available (`ollama pull {MODEL_NAME}`).\n\n"
            f"If using the HuggingFace fallback, ensure the model files are "
            f"downloaded and `OFFLINE_MODE` is set correctly."
        )
        return error_display, "*Error — no function calls completed.*", SAFETY_DISCLAIMER


def clear_session() -> tuple[str, str, str]:
    """Clear conversation history and reset UI panels."""
    global _orchestrator
    if _orchestrator is not None:
        _orchestrator.clear_history()
    return (
        "*History cleared. Upload an image or describe symptoms to begin.*",
        "*No analysis run yet.*",
        SAFETY_DISCLAIMER,
    )


# ─── UI construction ──────────────────────────────────────────────────────────

def build_demo():
    """
    Build and return the Gradio Blocks interface.
    Separated from main() to allow test-time import without launching a server.
    """
    import gradio as gr

    _CSS = """
    .disclaimer { background:#fff8e1; border:1px solid #f9a825;
                  border-radius:6px; padding:10px; font-size:0.88em; }
    .result-panel { line-height:1.65; }
    """

    with gr.Blocks(
        title="G4G RuralClinic AI — Dermatology Assistant",
        theme=gr.themes.Soft(),
        css=_CSS,
    ) as demo:

        gr.Markdown(
            "# G4G RuralClinic AI — Offline Dermatology Assistant\n"
            "**Powered by Gemma 4 · Fully Offline · Educational Use Only**\n\n"
            "Upload a skin lesion image and/or describe your symptoms. "
            "The AI will analyse the image using Gemma 4's multimodal capabilities "
            "and provide an informational assessment — not a diagnosis."
        )

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────────
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Skin Lesion Image",
                    height=280,
                )
                symptoms_input = gr.Textbox(
                    label="Symptom Description (optional)",
                    placeholder=(
                        "e.g. 'Dark mole on my upper arm that has been growing "
                        "and changing colour over the past 3 months. Sometimes itchy.'"
                    ),
                    lines=4,
                )
                offline_toggle = gr.Checkbox(
                    label="Offline Mode (disable internet access)",
                    value=OFFLINE_MODE,
                    info="All inference runs locally — no data leaves your machine.",
                )
                with gr.Row():
                    analyze_btn = gr.Button("Analyse", variant="primary", scale=3)
                    clear_btn = gr.Button("Clear", scale=1)

            # ── Right column: outputs ────────────────────────────────────
            with gr.Column(scale=1):
                result_output = gr.Markdown(
                    value="*Upload an image or enter symptoms to begin.*",
                    label="Analysis Result",
                    elem_classes=["result-panel"],
                )
                with gr.Accordion("Function Call Trace", open=False):
                    trace_output = gr.Markdown(
                        value="*No analysis run yet.*",
                    )

        disclaimer_box = gr.Markdown(
            value=SAFETY_DISCLAIMER,
            elem_classes=["disclaimer"],
        )

        # ── Wire callbacks ────────────────────────────────────────────────
        analyze_btn.click(
            fn=run_analysis,
            inputs=[image_input, symptoms_input, offline_toggle],
            outputs=[result_output, trace_output, disclaimer_box],
        )
        clear_btn.click(
            fn=clear_session,
            inputs=[],
            outputs=[result_output, trace_output, disclaimer_box],
        )

        gr.Markdown(
            "---\n"
            "**How it works**\n\n"
            "1. Your image + symptoms are processed entirely on your local machine.\n"
            "2. The orchestrator uses **Gemma 4 function calling** to decide "
            "   whether to invoke the dermatology specialist worker.\n"
            "3. The specialist analyses the image with Gemma 4's multimodal pipeline.\n"
            "4. Results are synthesised into plain language with mandatory safety guardrails.\n\n"
            "*G4G RuralClinic AI — bringing offline frontier AI to underserved communities.*"
        )

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    """Launch the Gradio demo server."""
    logger.info("Starting G4G RuralClinic AI demo on http://localhost:7860")
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,      # Offline-first: no public tunnel
        show_error=True,
    )


if __name__ == "__main__":
    main()
