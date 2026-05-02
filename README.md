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
| 3 | MedGemma worker, Docker/llama.cpp offline container, one-command startup, quantization support, low-end hardware testing | ✅ **Done** |
| 4 | Voice input, radiology worker, pharma/symptom-checker, Windows launcher | 🔜 In Progress |

---

## Quickstart

### 1. Install Ollama

Download from https://ollama.com, then pull the model:

```bash
ollama pull gemma4:e4b          # ~4B params, recommended (8+ GB RAM)
# ollama pull gemma4:e4b-q4_K_M # 4-bit quantized (~4 GB RAM, recommended for 4–8 GB)
# ollama pull gemma4:e2b-q4_K_M # 2-bit quantized (~2 GB RAM, for low-end laptops)
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

Or use the setup script (handles both steps above):

```bash
bash setup.sh
```

The setup script automatically detects system RAM and recommends an appropriate quantized model.

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

## Quantization Support

Quantized models (GGUF format via Ollama) significantly reduce memory usage while maintaining quality. This enables the system to run on laptops with 4–8 GB RAM.

| Model (Ollama tag) | RAM Estimate | Quality | Recommended for |
|---|---|---|---|
| `gemma4:e4b-q8_0`  | ~6 GB | High (8-bit) | Laptops with 8–16 GB RAM |
| `gemma4:e4b-q4_K_M` | ~4 GB | Medium-High (4-bit) | **Recommended** for 4–8 GB laptops |
| `gemma4:e4b` (default) | ~6 GB | High (~4-bit) | Laptops with 8+ GB RAM |
| `gemma4:e2b-q8_0`  | ~3 GB | Medium (8-bit) | Laptops with 2–4 GB RAM |
| `gemma4:e2b-q4_K_M` | ~2 GB | Medium (4-bit) | Ultra-low-end (<4 GB RAM) |

To use a quantized model, set the environment variable:

```bash
export USE_QUANTIZED=true
export QUANTIZED_MODEL_NAME=gemma4:e4b-q4_K_M
ollama pull gemma4:e4b-q4_K_M
```

The Orchestrator and DermatologyWorker automatically respect `USE_QUANTIZED` and `QUANTIZED_MODEL_NAME`.

---

## Docker / Docker Compose

### Quick start with Docker Compose (recommended)

```bash
# Copy example env
cp .env.example .env
# Edit .env if you want to change the quantized model

# Build and launch
docker compose up --build -d

# Access UI at http://localhost:7860

# View logs
docker compose logs -f app

# Stop
docker compose down
```

The compose file splits `ollama` and `app` services, with resource limits tuned for modest hardware. Volumes persist model weights and data between restarts.

### Standalone Docker run

```bash
# Build
docker build -t rural-clinic-ai .

# Run with quantized model (pulls on first run)
docker run -it --rm \
  -p 7860:7860 \
  -p 11434:11434 \
  -e QUANTIZED_MODEL_NAME=gemma4:e4b-q4_K_M \
  -e USE_QUANTIZED=true \
  -v ./ollama_data:/root/.ollama \
  -v ./data:/app/data \
  rural-clinic-ai
```

The `docker-entrypoint.sh` script manages starting Ollama, pulling the model, and launching the Gradio UI.

---

## Low-End Hardware Testing

Simulate constrained resources and benchmark inference times:

```bash
# Test with 4 GB RAM / 2 CPU limit
bash scripts/test_low_end.sh --memory 4096 --cpus 2.0

# Compare all quantized variants
bash scripts/test_low_end.sh --compare

# Custom
bash scripts/test_low_end.sh --model gemma4:e4b-q4_K_M --samples 5 --timeout 120
```

Results are saved to `logs/low_end_benchmark.json`. The script tests image analysis and RAG query latency under resource constraints.

#### Performance targets (on 8 GB laptop)
- **gemma4:e4b-q4_K_M**: 8–20s per image analysis (typical)
- **gemma4:e2b-q8_0**: 4–10s per image analysis (typical)
- **RAG queries**: 500–2000ms (typical)

---

## Hardware Requirements Table (Quantized vs Non-Quantized)

| Configuration | Model | RAM Required | Typical Image Analysis Latency | Notes |
|---|---|---|---|---|
| **Non-quantized (Ollama default)** | gemma4:e4b | ~6 GB | 8–20s CPU | Best quality, recommended for 8+ GB |
| **Quantized 4-bit** | gemma4:e4b-q4_K_M | ~4 GB | 8–20s CPU | Best balance for 4–8 GB laptops |
| **Quantized 8-bit** | gemma4:e4b-q8_0 | ~6 GB | 8–18s CPU | Slightly faster, similar RAM |
| **Lightweight 2B** | gemma4:e2b-q4_K_M | ~2 GB | 4–10s CPU | Good for <4 GB systems |
| **Lightweight 2B-8bit** | gemma4:e2b-q8_0 | ~3 GB | 4–10s CPU | Fast, low RAM |

Latency varies by CPU speed, image complexity, and system load. All tests assume CPU-only inference (no GPU).

---

## Safety & Plain-Language Enhancements

All outputs pass through enhanced safety guardrails:
- Overconfident diagnostic language is replaced with appropriately hedged phrases.
- Referral timing is stated clearly but compassionately.
- Plain-language explanations avoid jargon for users with low health literacy.
- Mandatory medical disclaimer appended to every response.

The safety module enforces:
- No definitive diagnoses ("appears consistent with" vs "this is").
- Compassionate, accessible wording.
- Clear, calm referral guidance ("seek medical care" vs "go to ER immediately").

---

## Evaluation

Run a dry-run (no Ollama required):

```bash
python -m eval.evaluate --mock
```

Run on real ISIC-format samples:

```bash
python -m eval.evaluate --data-dir data/samples --max-samples 200
```

Results are printed as a P/R/F1 table and saved to `logs/eval_results.json`.

---

## Running tests

```bash
python -m pytest tests/ -v
```

133 tests across three test files — all pass with mocked Ollama (no local server needed):

| File | Coverage |
|---|---|
| `tests/test_dermatology.py` | DermatologyWorker: init, preprocessing, analysis structure, JSON parsing, safety, all 10 classes |
| `tests/test_orchestrator.py` | Orchestrator: init, routing, multi-tool dispatch, history, error resilience |
| `tests/test_rag.py` | KnowledgeRetriever: TF-IDF index, retrieval, edge cases; RAGWorker: answer structure, fallbacks; Evaluator: metrics |

---

## Project structure

```
src/
  config.py              — model names, paths, safety constants, RAG config, quantization settings
  app/demo.py            — Gradio web UI (two tabs: Image Analysis + Knowledge Chat)
  orchestrator/core.py   — multi-tool routing, response synthesis, history
  workers/
    dermatology.py       — Gemma 4 multimodal skin lesion analysis worker
    rag_worker.py        — RAG Q&A worker (retriever + Gemma 4 synthesis)
  rag/
    retriever.py         — offline TF-IDF retriever (numpy only)
    knowledge_base.json  — 34 curated dermatology knowledge chunks
  tools/__init__.py      — Ollama function-call schemas (analyze + RAG)
  utils/safety.py        — disclaimer enforcement, language softening, compassionate wording
docker/
  Dockerfile             — multi-stage build: Ollama + Python app
  docker-compose.yml     — orchestrates ollama + app services
  docker-entrypoint.sh   — manages server start, model pull, and UI launch
scripts/
  download_sample_data.sh— fetch or generate sample dermatology images
  test_low_end.sh        — simulate constrained resources + run benchmarks
eval/
  evaluate.py            — evaluation harness (ISIC dir, ISIC CSV, --mock mode)
tests/
  test_dermatology.py
  test_orchestrator.py
  test_rag.py
```

---

## Roadmap detail

### Step 3 — Offline container + quantization (✅ Done)
- ✅ Dockerfile (multi-stage) with Ollama + Python app
- ✅ docker-compose.yml for local testing with model/data volumes
- ✅ CPU-only mode support for modest hardware
- ✅ Quantization support: gemma4:e4b-q4_K_M, e4b-q8_0, e2b-q4_K_M, etc.
- ✅ `--quantized` flag / `USE_QUANTIZED` env var for model switching
- ✅ scripts/test_low_end.sh for constrained-resource simulation
- ✅ Updated safety.py with enhanced hedging, compassion, referral timing
- ✅ Plain-language output optimized for low health literacy

### Step 4 — Multi-domain + voice (🔜 In Progress)
- MedGemma worker replacement for dermatology
- Radiology worker (MedGemma on CXR-like data)
- Pharma/symptom-checker worker
- Voice input via Gemma 4 native audio support
- Structured triage JSON output with `<|think|>` reasoning trace visible in UI

---

**This tool is not a substitute for professional medical advice. All outputs carry a mandatory disclaimer and are for informational purposes only.**

---

**This tool is not a substitute for professional medical advice. All outputs carry a mandatory disclaimer and are for informational purposes only.**
