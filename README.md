# G4G RuralClinic AI

Offline multimodal dermatology screening assistant powered by Gemma 4. Designed for rural and underserved communities where internet access is unreliable. Runs fully on a standard laptop (8–16 GB RAM, CPU-only).

---

## Quickstart

### 1. Install Ollama

Download and install from https://ollama.com, then pull the model:

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

### 3. Download sample images (optional)

```bash
bash scripts/download_sample_data.sh
```

### 4. Launch the demo

```bash
ollama serve                    # start Ollama in one terminal
python -m src.app.demo          # launch the Gradio UI in another
```

Open http://localhost:7860 in your browser.

---

## Offline use

After the initial `ollama pull`, no internet is required:

```bash
export OFFLINE_MODE=true
ollama serve
python -m src.app.demo
```

---

## Hardware notes

| Variant | RAM required | Notes |
|---|---|---|
| `gemma4:e4b` (Ollama) | ~6 GB | Recommended — 4-bit quantised |
| `gemma4:e2b` (Ollama) | ~3 GB | Low-end laptops |
| `google/gemma-4-e4b-it` (HuggingFace) | ~12–16 GB | CPU float32; use Ollama instead |

---

## Running tests

```bash
python -m pytest tests/ -v
```

---

## Project structure

```
src/
  config.py          — model names, paths, safety constants
  app/demo.py        — Gradio web UI
  orchestrator/      — routing and response synthesis
  workers/           — dermatology specialist worker
  tools/             — Ollama function-call schemas
  utils/safety.py    — medical disclaimer enforcement
scripts/
  download_sample_data.sh
tests/
```

---

**This tool is not a substitute for professional medical advice. All outputs carry a mandatory disclaimer and are for informational purposes only.**
