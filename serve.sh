#!/bin/bash
# Start the vLLM OpenAI-compatible API server.
# Run once in a tmux/screen session — the model stays loaded in GPU memory.
# The frontend connects to it when LLM_BACKEND=vllm is set in .env.
#
# Usage:
#   tmux new -s llm
#   bash serve.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

source .venv/bin/activate

# Load project env vars (HF_HOME, VLLM_MODEL, etc.)
set -o allexport
source .env
set +o allexport

# RTX 5090 is at PCI bus index 1 (index 0 is the 2080 Ti, which is excluded).
# Override with VLLM_GPU= in .env if your setup differs.
MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-14B-Instruct-AWQ}"
PORT="${VLLM_PORT:-8000}"
GPU="${VLLM_GPU:-1}"

# RTX 5090 has 32 GB VRAM. With 14B AWQ (~8 GB weights) the remaining
# ~24 GB goes to KV cache — enough for ~15 concurrent sequences safely.
MAX_SEQS="${VLLM_MAX_SEQS:-15}"

echo "Starting vLLM server"
echo "  Model     : $MODEL"
echo "  Port      : $PORT"
echo "  GPU       : cuda:$GPU  (RTX 5090)"
echo "  Max seqs  : $MAX_SEQS  (concurrent inference slots)"
echo "  Cache     : ${HF_HOME:-~/.cache/huggingface}"
echo ""

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$GPU" \
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --quantization awq \
    --dtype float16 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 8192 \
    --max-num-seqs "$MAX_SEQS" \
    --served-model-name "$MODEL" \
    --trust-remote-code \
    --no-enable-log-requests
