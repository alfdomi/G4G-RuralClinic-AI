# G4G RuralClinic AI

Offline multimodal dermatology screening assistant powered by Gemma 4.
Designed for rural and underserved communities where internet access is unreliable.
Runs fully on a standard laptop (8–16 GB RAM, CPU-only).

---

## Build Status

| Step | Feature | Status |
|---|---|---|
| 1 | Core orchestrator + DermatologyWorker + Gradio shell | ✅ Done |
| 2 | Full routing, RAG tools, Gradio two-tab UI, evaluation harness | ✅ Done |
| 3 | MedGemma worker, Docker/llama.cpp offline container, one-command startup | 🔜 Next |
| 4 | Voice input, radiology worker, pharma/symptom-checker, Windows launcher | 🔜 Planned |

---

## Architecture

```
User (image + symptoms / voice)
        │
        ▼
┌──────────────────────────────────────────┐
│   Orchestrator  (Gemma 4 E4B)            │
│   • Native function calling              │
│   • <|think|> triage reasoning           │
│   • Outputs structured JSON routing      │
│     { domain, triage_level,              │
│       extracted_symptoms, next_worker }  │
└──────────┬───────────────────────────────┘
           │ routes to one or more workers
    ┌──────┴──────────────────────────────┐
    │                                     │
    ▼                                     ▼
┌────────────────────┐      ┌────────────────────────┐
│ DermatologyWorker  │      │      RAGWorker          │
│ Gemma 4 multimodal │      │  TF-IDF retriever over  │
│ Image + ABCDE      │      │  offline knowledge base │
│ Classification     │      │  (34 derm. chunks)      │
│ Risk triage        │      │  + Gemma 4 synthesis    │
└────────────────────┘      └────────────────────────┘
           │                             │
           └──────────────┬──────────────┘
                          ▼
              ┌─────────────────────┐
              │  Safety Enforcer    │
              │  • Soften overconf. │
              │  • Append disclaimer│
              └─────────────────────┘
                          │
                          ▼
              Plain-language response
              + structured JSON result
              + function-call trace
```

**Planned additions (Step 3–4):**
- Replace DermatologyWorker backbone with **MedGemma 1.5 4B** (purpose-built for dermatology/medical imaging)
- Add **RadiologyWorker** (MedGemma on CXR-like data)
- Add **PharmaWorker** (symptom-checker + medication lookup)
- **Docker container** with Ollama + GGUF quants + Chroma/FAISS RAG — one-command startup
- **llama.cpp** server fallback for ultra-low-resource hardware
- **Voice input** (Gemma 4 native audio support)

---

## Quickstart

### 1. Install Ollama

Download from https://ollama.com, then pull the model:

```bash
ollama pull gemma4:e4b          # ~4B params, recommended (8+ GB RAM)
# ollama pull gemma4:e2b        # lighter alternative for <8 GB RAM
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

Or use the setup script (handles both steps above):

```bash
bash setup.sh
```

### 3. Download sample images (optional — for evaluation)

```bash
bash scripts/download_sample_data.sh
```

### 4. Launch the demo

```bash
ollama serve                    # start Ollama in one terminal
python -m src.app.demo          # launch the Gradio UI in another
```

Open http://localhost:7860 in your browser.

The UI has two tabs:
- **Image Analysis** — upload a skin lesion image and/or describe symptoms; Gemma 4 function calling dispatches to the DermatologyWorker and returns a structured ABCDE assessment.
- **Knowledge Chat** — ask plain-language dermatology questions; the RAG worker retrieves from the offline knowledge base and synthesises a grounded answer.

---

## Offline use

After the initial `ollama pull`, no internet is required:

```bash
export OFFLINE_MODE=true
ollama serve
python -m src.app.demo
```

The **Offline Mode** checkbox in the UI header also controls this at runtime.

---

## Running tests

```bash
python -m pytest tests/ -v
```

98 tests across three test files — all pass with mocked Ollama (no local server needed):

| File | Coverage |
|---|---|
| `tests/test_dermatology.py` | DermatologyWorker: init, preprocessing, analysis structure, JSON parsing, safety, all 10 classes |
| `tests/test_orchestrator.py` | Orchestrator: init, routing, multi-tool dispatch, history, error resilience |
| `tests/test_rag.py` | KnowledgeRetriever: TF-IDF index, retrieval, edge cases; RAGWorker: answer structure, fallbacks; Evaluator: metrics |

---

## Evaluation

Run a dry-run (no Ollama required):

```bash
python -m eval.evaluate --mock
```

Run on real ISIC-format samples (directory tree: `data/samples/<class>/*.jpg`):

```bash
python -m eval.evaluate --data-dir data/samples --max-samples 200
```

ISIC CSV format:

```bash
python -m eval.evaluate --data-dir data/isic --csv data/isic/labels.csv
```

Results are printed as a P/R/F1 table and saved to `logs/eval_results.json`.

---

## Hardware notes

| Variant | RAM required | Notes |
|---|---|---|
| `gemma4:e4b` (Ollama) | ~6 GB | Recommended — 4-bit quantised |
| `gemma4:e2b` (Ollama) | ~3 GB | Low-end laptops |
| `google/gemma-4-e4b-it` (HuggingFace) | ~12–16 GB | CPU float32; use Ollama instead |
| MedGemma 1.5 4B (planned) | ~4–6 GB | Medical imaging specialist worker |

---

## Project structure

```
src/
  config.py              — model names, paths, safety constants, RAG config
  app/demo.py            — Gradio web UI (two tabs: Image Analysis + Knowledge Chat)
  orchestrator/core.py   — multi-tool routing, response synthesis, history
  workers/
    dermatology.py       — Gemma 4 multimodal skin lesion analysis worker
    rag_worker.py        — RAG Q&A worker (retriever + Gemma 4 synthesis)
  rag/
    retriever.py         — offline TF-IDF retriever (numpy only)
    knowledge_base.json  — 34 curated dermatology knowledge chunks
  tools/__init__.py      — Ollama function-call schemas (analyze + RAG)
  utils/safety.py        — disclaimer enforcement, language softening
eval/
  evaluate.py            — evaluation harness (ISIC dir, ISIC CSV, --mock mode)
scripts/
  download_sample_data.sh
tests/
  test_dermatology.py
  test_orchestrator.py
  test_rag.py
```

---

## Roadmap detail

### Step 3 — Offline container + MedGemma (next)
- Swap DermatologyWorker backbone to MedGemma 1.5 4B via Ollama GGUF
- Embed dermatology guidelines (WHO, Fitzpatrick) into Chroma/FAISS vector store
- Dockerfile: Ollama + app + RAG store, `docker compose up` launches everything
- llama.cpp server fallback for < 8 GB RAM targets
- Windows `.bat` / PowerShell launcher for non-technical clinic staff

### Step 4 — Multi-domain + voice
- Radiology worker (MedGemma on CXR-like data + basic enhancement)
- Pharma/symptom-checker worker (Gemma 4 + medication/interaction lookup)
- Voice input via Gemma 4's native audio support
- Structured triage JSON output with `<|think|>` reasoning trace visible in UI

---

**This tool is not a substitute for professional medical advice. All outputs carry a mandatory disclaimer and are for informational purposes only.**
