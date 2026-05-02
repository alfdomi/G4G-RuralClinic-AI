#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────────
# Low-End Hardware Simulation and Benchmark Script
#
# Simulates constrained hardware environments (limited RAM/CPU) and benchmarks
# Gemma 4 inference times across different quantization levels for dermatology
# image analysis and RAG queries.
#
# Usage:
#   bash scripts/test_low_end.sh [OPTIONS]
#
# Options:
#   --memory MB        Virtual memory limit (e.g., 2048 for 2 GB) [default: 8192]
#   --cpus N           CPU cores limit (fractional allowed) [default: 2.0]
#   --timeout SEC      Max seconds per benchmark [default: 120]
#   --model MODEL      Quantized model to test [default: gemma4:e4b-q4_K_M]
#   --samples N        Number of images to test [default: 5]
#   --compare          Test multiple quantization levels and compare
#   --help             Show this help
#
# Examples:
#   # Test on 4 GB RAM / 2 CPU
#   bash scripts/test_low_end.sh --memory 4096 --cpus 2.0
#
#   # Compare all quantized variants
#   bash scripts/test_low_end.sh --compare
#
# Prerequisites:
#   - Ollama running locally (or in Docker)
#   - Model(s) already pulled via `ollama pull <model>`
#   - Sample test images in data/samples/
#
# Output:
#   - Console report with latency (ms) and RAM usage
#   - JSON log at logs/low_end_benchmark.json
#
# Notes:
#   - Uses cgroups / ulimit where available to simulate constraints.
#   - Falls back to nice/ionice if hard limits unavailable.
#   - For Docker-based constraints, use docker-compose resource limits.
#
# ────────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$ROOT_DIR/logs"
SAMPLE_DIR="$ROOT_DIR/data/samples"
RESULTS_FILE="$LOG_DIR/low_end_benchmark.json"

# ── Default parameters ───────────────────────────────────────────────────────
MEMORY_MB=${MEMORY_MB:-8192}
CPUS=${CPUS:-2.0}
TIMEOUT_SEC=${TIMEOUT_SEC:-120}
MODEL=${MODEL:-gemma4:e4b-q4_K_M}
SAMPLES=${SAMPLES:-5}
COMPARE=${COMPARE:-false}

# Parse CLI arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --memory) MEMORY_MB="$2"; shift 2 ;;
        --cpus)   CPUS="$2"; shift 2 ;;
        --timeout) TIMEOUT_SEC="$2"; shift 2 ;;
        --model)  MODEL="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --compare) COMPARE=true; shift ;;
        --help)   cat "$0" | head -40; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mkdir -p "$LOG_DIR"

# ── Helper functions ─────────────────────────────────────────────────────────
log_info()  { echo "[$(date +%H:%M:%S)] [INFO]  $*"; }
log_warn()  { echo "[$(date +%H:%M:%S)] [WARN]  $*" >&2; }
log_error() { echo "[$(date +%H:%M:%S)] [ERROR] $*" >&2; }

# Check prerequisites
check_prereqs() {
    local ok=true
    if ! command -v ollama &> /dev/null; then
        log_error "Ollama not found. Install from https://ollama.com"
        ok=false
    fi
    if ! command -v python3 &> /dev/null; then
        log_error "python3 not found"
        ok=false
    fi
    if [ ! -d "$SAMPLE_DIR" ] || [ -z "$(ls -A "$SAMPLE_DIR" 2>/dev/null)" ]; then
        log_warn "No sample images in $SAMPLE_DIR"
        log_info "Run: bash scripts/download_sample_data.sh"
        # Continue anyway; some tests may work without images
    fi
    if ! curl -s -f "http://localhost:11434/api/tags" > /dev/null 2>&1; then
        log_error "Ollama not running. Start it with: ollama serve"
        ok=false
    fi
    if ! $ok; then
        exit 1
    fi
}

# Check if a model is available locally
model_available() {
    local m="$1"
    curl -s "http://localhost:11434/api/tags" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = [mdl['name'] for mdl in data.get('models', [])]
    # Match with or without :latest suffix
    for model in models:
        if model.startswith('$m:'):
            sys.exit(0)
        if model == '$m' or model == '$m:latest':
            sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
" 2>/dev/null
}

# Apply resource constraints using available mechanisms
apply_constraints() {
    local mem_mb="$1"
    local cpus="$2"
    local prefix="$3"  # command to prefix

    # Build a constrained execution command
    local cmd=""

    # Method 1: systemd-run (most reliable for cgroup limits)
    if command -v systemd-run &> /dev/null && systemd-run --version &> /dev/null; then
        cmd="systemd-run --user --scope -p MemoryLimit=${mem_mb}M -p CPUQuota=${cpus}00%"
    # Method 2: ulimit + cpulimit
    elif command -v cpulimit &> /dev/null; then
        local cpu_pct=$(echo "$cpus * 100" | bc | cut -d. -f1)
        cmd="ulimit -v $((mem_mb * 1024)) && cpulimit -l $cpu_pct --"
    # Method 3: ulimit only
    else
        cmd="ulimit -v $((mem_mb * 1024)) 2>/dev/null || true;"
    fi

    echo "$cmd"
}

# Run a Python benchmark script with constraints
run_benchmark() {
    local model="$1"
    local script="$2"
    local desc="$3"
    local mem_mb="$4"
    local cpus="$5"
    local timeout_sec="$6"

    log_info "Testing: $desc (model=$model, mem=${mem_mb}MB, cpus=$cpus)"

    local const_cmd
    const_cmd=$(apply_constraints "$mem_mb" "$cpus" "python3")

    local full_cmd
    if [ -n "$const_cmd" ] && [ "$const_cmd" != "true" ]; then
        full_cmd="$const_cmd timeout ${timeout_sec}s python3 $script --model '$model' 2>&1"
    else
        full_cmd="timeout ${timeout_sec}s python3 $script --model '$model' 2>&1"
    fi

    local start_time
    start_time=$(date +%s%N)
    local output
    local exit_code=0

    output=$(eval "$full_cmd" 2>&1) || exit_code=$?

    local end_time
    end_time=$(date +%s%N)
    local elapsed_ms=$(( (end_time - start_time) / 1000000 ))

    if [ $exit_code -eq 124 ]; then
        log_error "Timeout after ${timeout_sec}s: $desc"
        elapsed_ms=$(( timeout_sec * 1000 ))
    elif [ $exit_code -ne 0 ]; then
        log_error "Benchmark failed (exit $exit_code): $desc"
        log_error "Output: $output"
    else
        log_info "Completed in ${elapsed_ms}ms: $desc"
    fi

    # Extract memory info if available
    local max_rss_kb=0
    if echo "$output" | grep -q "Maximum resident"; then
        max_rss_kb=$(echo "$output" | grep -oP '\d+(?= kilobytes)')
    fi

    echo "${elapsed_ms}|${max_rss_kb}|${exit_code}|${model}|${desc}"
}

# ── Benchmark: Image analysis via Python ─────────────────────────────────────
benchmark_image_analysis() {
    local model="$1"
    local mem_mb="$2"
    local cpus="$3"
    local timeout_sec="$4"
    local num_samples="$5"

    # Create inline Python benchmark script
    local py_script
    py_script=$(mktemp /tmp/bench_img_XXXXXX.py)
    cat > "$py_script" << 'PYEOF'
import os, sys, time, json
from PIL import Image
import random

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ["OFFLINE_MODE"] = "true"
os.environ["USE_QUANTIZED"] = "true"
os.environ["QUANTIZED_MODEL_NAME"] = sys.argv[2] if len(sys.argv) > 2 else "gemma4:e4b-q4_K_M"

from src.orchestrator.core import Orchestrator
from src.config import MODEL_NAME

def main():
    model_name = sys.argv[2] if len(sys.argv) > 2 else MODEL_NAME
    num_samples = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    print(f"Initializing Orchestrator with model={model_name}...")
    orch = Orchestrator(use_ollama=True)

    # Load sample images
    sample_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'samples')
    images = []
    if os.path.isdir(sample_dir):
        for fname in sorted(os.listdir(sample_dir)):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = Image.open(os.path.join(sample_dir, fname))
                    images.append(img)
                    if len(images) >= num_samples:
                        break
                except Exception as e:
                    print(f"  Skipping {fname}: {e}")

    if not images:
        print("No sample images found. Using generated dummy image.")
        images = [Image.new('RGB', (224, 224), color=(random.randint(0,255), random.randint(0,255), random.randint(0,255)))]

    print(f"Running {len(images)} image analysis benchmarks...")
    latencies = []
    for i, img in enumerate(images):
        start = time.time()
        try:
            result = orch.route_and_run(
                query="Please analyze this skin lesion.",
                image=img
            )
            elapsed = (time.time() - start) * 1000
            latencies.append(elapsed)
            cls = result.get('worker_result', {}).get('classification', 'N/A')
            print(f"  Sample {i+1}: {elapsed:.0f}ms | {cls}")
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            latencies.append(elapsed)
            print(f"  Sample {i+1}: {elapsed:.0f}ms | ERROR: {e}")

    if latencies:
        avg = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies)//2]
        p95 = sorted(latencies)[int(len(latencies)*0.95)] if len(latencies) > 1 else latencies[0]
        print(f"\nSummary:")
        print(f"  Avg: {avg:.0f}ms")
        print(f"  P50: {p50:.0f}ms")
        print(f"  P95: {p95:.0f}ms")
        print(f"  Min: {min(latencies):.0f}ms")
        print(f"  Max: {max(latencies):.0f}ms")

if __name__ == "__main__":
    main()
PYEOF

    # Get memory stats before and after
    local result
    result=$(run_benchmark "$model" "$py_script" "image_analysis" "$mem_mb" "$cpus" "$timeout_sec")

    rm -f "$py_script"
    echo "$result"
}

# ── Benchmark: RAG query ─────────────────────────────────────────────────────
benchmark_rag() {
    local model="$1"
    local mem_mb="$2"
    local cpus="$3"
    local timeout_sec="$4"

    local py_script
    py_script=$(mktemp /tmp/bench_rag_XXXXXX.py)
    cat > "$py_script" << 'PYEOF'
import os, sys, time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

os.environ["OFFLINE_MODE"] = "true"
os.environ["USE_QUANTIZED"] = "true"
os.environ["QUANTIZED_MODEL_NAME"] = sys.argv[2] if len(sys.argv) > 2 else "gemma4:e4b-q4_K_M"

from src.workers.rag import RAGWorker

def main():
    model_name = sys.argv[2] if len(sys.argv) > 2 else "gemma4:e4b-q4_K_M"
    print(f"Initializing RAGWorker with model={model_name}...")
    rag = RAGWorker(use_ollama=True)

    test_queries = [
        "What are the ABCDE criteria for melanoma detection?",
        "How does eczema present differently on darker skin tones?",
        "When should a patient with a suspicious mole be referred urgently?",
    ]

    print(f"Running {len(test_queries)} RAG query benchmarks...")
    latencies = []
    for i, q in enumerate(test_queries):
        start = time.time()
        try:
            result = rag.answer(q)
            elapsed = (time.time() - start) * 1000
            latencies.append(elapsed)
            print(f"  Query {i+1}: {elapsed:.0f}ms")
        except Exception as e:
            elapsed = (time.time() - start) * 1000
            latencies.append(elapsed)
            print(f"  Query {i+1}: {elapsed:.0f}ms | ERROR: {e}")

    if latencies:
        avg = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies)//2]
        print(f"\nSummary:")
        print(f"  Avg: {avg:.0f}ms")
        print(f"  P50: {p50:.0f}ms")

if __name__ == "__main__":
    main()
PYEOF

    local result
    result=$(run_benchmark "$model" "$py_script" "rag_query" "$mem_mb" "$cpus" "$timeout_sec")

    rm -f "$py_script"
    echo "$result"
}

# ── Compare multiple quantization levels ──────────────────────────────────────
compare_quantization_levels() {
    local mem_mb="$1"
    local cpus="$2"
    local timeout_sec="$3"
    local num_samples="$4"

    local models=(
        "gemma4:e4b-q8_0:~6GB"
        "gemma4:e4b-q4_K_M:~4GB"
        "gemma4:e4b-q4_0:~3.8GB"
        "gemma4:e2b-q8_0:~3GB"
        "gemma4:e2b-q4_K_M:~2GB"
    )

    log_info "=== QUANTIZATION COMPARISON ==="
    log_info "Hardware limit: ${mem_mb}MB RAM, ${cpus} CPUs"
    log_info ""

    local results_json="[]"
    local all_results=""

    for entry in "${models[@]}"; do
        local m="${entry%%:*}"
        local info="${entry#*:}"

        if ! model_available "$m"; then
            log_warn "Model $m not available locally. Skipping."
            log_info "  Pull with: ollama pull $m"
            continue
        fi

        log_info "─────────────────────────────────"
        log_info "Testing: $m ($info)"
        log_info "─────────────────────────────────"

        # Image analysis
        local img_result
        img_result=$(benchmark_image_analysis "$m" "$mem_mb" "$cpus" "$timeout_sec" "$num_samples")
        IFS='|' read -r ms rss ec md desc <<< "$img_result"
        all_results="${all_results}\n${md} | ${desc} | ${ms}ms | ${rss}KB RSS"

        # RAG query
        local rag_result
        rag_result=$(benchmark_rag "$m" "$mem_mb" "$cpus" "$timeout_sec")
        IFS='|' read -r ms2 rss2 ec2 md2 desc2 <<< "$rag_result"
        all_results="${all_results}\n${md2} | ${desc2} | ${ms2}ms"

        log_info ""
    done

    if [ -n "$all_results" ]; then
        echo ""
        log_info "=== SUMMARY TABLE ==="
        printf "%-30s %-20s %10s %12s\n" "Model" "Benchmark" "Latency" "Memory"
        printf "%-30s %-20s %10s %12s\n" "-----" "---------" "------" "-------"
        echo -e "$all_results" | grep -v '^$' | while IFS='|' read -r model bench latency mem; do
            printf "%-30s %-20s %10s %12s\n" "$model" "$bench" "${latency}" "${mem}"
        done
    fi
}

# ── Main execution ────────────────────────────────────────────────────────────
main() {
    echo ""
    echo "══════════════════════════════════════════════════════════════════════════"
    echo "  G4G RuralClinic AI — Low-End Hardware Benchmark"
    echo "══════════════════════════════════════════════════════════════════════════"
    echo ""

    check_prereqs

    if [ "$COMPARE" = true ]; then
        compare_quantization_levels "$MEMORY_MB" "$CPUS" "$TIMEOUT_SEC" "$SAMPLES"
        exit 0
    fi

    # Single model benchmark
    if ! model_available "$MODEL"; then
        log_error "Model '$MODEL' not available locally."
        log_info "Pull with: ollama pull $MODEL"
        exit 1
    fi

    log_info "Starting benchmark (single model mode)"
    log_info "  Model:  $MODEL"
    log_info "  Memory: ${MEMORY_MB}MB limit"
    log_info "  CPUs:   $CPUS limit"
    log_info "  Timeout: ${TIMEOUT_SEC}s per test"
    log_info ""

    # Image analysis benchmark
    log_info "─────────────────────────────────────────"
    log_info "IMAGE ANALYSIS BENCHMARK"
    log_info "─────────────────────────────────────────"
    img_result=$(benchmark_image_analysis "$MODEL" "$MEMORY_MB" "$CPUS" "$TIMEOUT_SEC" "$SAMPLES")
    IFS='|' read -r ms rss ec md desc <<< "$img_result"

    # RAG benchmark
    log_info "─────────────────────────────────────────"
    log_info "RAG QUERY BENCHMARK"
    log_info "─────────────────────────────────────────"
    rag_result=$(benchmark_rag "$MODEL" "$MEMORY_MB" "$CPUS" "$TIMEOUT_SEC")
    IFS='|' read -r ms2 rss2 ec2 md2 desc2 <<< "$rag_result"

    # Save results as JSON
    cat > "$RESULTS_FILE" << RESULTJSON
{
  "timestamp": "$(date -Iseconds)",
  "config": {
    "model": "$MODEL",
    "memory_limit_mb": ${MEMORY_MB},
    "cpu_limit": ${CPUS},
    "timeout_sec": ${TIMEOUT_SEC},
    "num_samples": ${SAMPLES}
  },
  "results": {
    "image_analysis": {
      "latency_ms": ${ms:-0},
      "memory_rss_kb": ${rss:-0},
      "exit_code": ${ec:-0},
      "description": "$desc"
    },
    "rag_query": {
      "latency_ms": ${ms2:-0},
      "memory_rss_kb": ${rss2:-0},
      "exit_code": ${ec2:-0},
      "description": "$desc2"
    }
  }
}
RESULTJSON

    log_info "Results saved to: $RESULTS_FILE"
    echo ""
}

main
