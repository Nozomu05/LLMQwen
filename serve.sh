#!/bin/bash
# Start the full RAG stack:
#   1. vLLM server       (GPU 1 - RTX 5090,  port 8000)
#   2. LLM service       (CPU,                port 8012)
#   3. Reranker service  (GPU 0 - RTX 2080Ti, port 8011)
#   4. Search service    (GPU 0 - RTX 2080Ti, port 8010)
#
# Each process runs in its own tmux pane so you can monitor logs separately.
# Requires tmux. Run with:
#   bash serve.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

MODEL="${VLLM_MODEL:-Qwen/Qwen3-14B-AWQ}"
VLLM_PORT="${VLLM_PORT:-8000}"
GPU="${VLLM_GPU:-1}"
MAX_SEQS="${VLLM_MAX_SEQS:-15}"

LLM_SVC_PORT="${LLM_SERVICE_PORT:-8012}"
RERANKER_SVC_PORT="${RERANKER_SERVICE_PORT:-8011}"
SEARCH_SVC_PORT="${SEARCH_SERVICE_PORT:-8010}"

SESSION="rag"

# Kill any existing session with the same name
tmux kill-session -t "$SESSION" 2>/dev/null || true

echo "Starting RAG stack in tmux session '$SESSION' ..."
echo ""

# ── Window 0: vLLM server ──────────────────────────────────────────────────
tmux new-session -d -s "$SESSION" -n "vllm"
tmux send-keys -t "$SESSION:vllm" "
cd '$SCRIPT_DIR'
source .venv/bin/activate
echo '=== vLLM server (GPU $GPU - RTX 5090, port $VLLM_PORT) ==='
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES='$GPU' \\
python -m vllm.entrypoints.openai.api_server \\
    --model '$MODEL' \\
    --port $VLLM_PORT \\
    --tensor-parallel-size 1 \\
    --quantization awq \\
    --dtype float16 \\
    --gpu-memory-utilization 0.75 \\
    --max-model-len 8192 \\
    --max-num-seqs $MAX_SEQS \\
    --served-model-name '$MODEL' \\
    --trust-remote-code \\
    --no-enable-log-requests \\
    --enable-chunked-prefill \\
    --enable-prefix-caching
" Enter

# ── Window 1: LLM service ─────────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "llm-svc"
tmux send-keys -t "$SESSION:llm-svc" "
cd '$SCRIPT_DIR'
source .venv/bin/activate
set -o allexport; source services/llm/.env; set +o allexport
echo '=== LLM service (port $LLM_SVC_PORT) ==='
uvicorn services.llm.app:app --host 0.0.0.0 --port $LLM_SVC_PORT
" Enter

# ── Window 2: Reranker service ────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "reranker-svc"
tmux send-keys -t "$SESSION:reranker-svc" "
cd '$SCRIPT_DIR'
source .venv/bin/activate
set -o allexport; source services/reranker/.env; set +o allexport
echo '=== Reranker service (GPU 0 - RTX 2080 Ti, port $RERANKER_SVC_PORT) ==='
CUDA_DEVICE_ORDER=PCI_BUS_ID \\
uvicorn services.reranker.app:app --host 0.0.0.0 --port $RERANKER_SVC_PORT
" Enter

# ── Window 3: Search service ──────────────────────────────────────────────
tmux new-window -t "$SESSION" -n "search-svc"
tmux send-keys -t "$SESSION:search-svc" "
cd '$SCRIPT_DIR'
source .venv/bin/activate
set -o allexport; source services/search/.env; set +o allexport
echo '=== Search service (GPU 0 - RTX 2080 Ti, port $SEARCH_SVC_PORT) ==='
CUDA_DEVICE_ORDER=PCI_BUS_ID \\
uvicorn services.search.app:app --host 0.0.0.0 --port $SEARCH_SVC_PORT
" Enter

echo "All services starting in tmux session '$SESSION'."
echo ""
echo "Attach with:  tmux attach -t $SESSION"
echo "Switch panes: Ctrl-b then 0=vllm  1=llm-svc  2=reranker-svc  3=search-svc"
echo ""
echo "Service URLs:"
echo "  vLLM server      : http://localhost:$VLLM_PORT"
echo "  LLM service      : http://localhost:$LLM_SVC_PORT"
echo "  Reranker service : http://localhost:$RERANKER_SVC_PORT"
echo "  Search service   : http://localhost:$SEARCH_SVC_PORT"
