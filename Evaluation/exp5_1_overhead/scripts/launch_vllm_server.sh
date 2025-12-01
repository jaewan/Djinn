#!/bin/bash
# Launch vLLM server for Experiment 5.1 baselines
#
# This script starts a vLLM OpenAI-compatible server that can be used
# with run_vllm_client.py to measure baseline LLM serving performance.
#
# Prerequisites:
#   pip install vllm>=0.4.0
#
# Usage:
#   ./launch_vllm_server.sh                          # Uses defaults
#   MODEL_ID=meta-llama/Llama-2-7b-hf ./launch_vllm_server.sh
#   PORT=8001 ./launch_vllm_server.sh

set -e

# Configuration (can be overridden via environment)
MODEL_ID="${MODEL_ID:-sshleifer/tiny-gpt2}"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"  # Reduced from 0.9 to avoid OOM if Djinn is collocated. See: OSDI Review - Code Nit #2
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

echo "=========================================="
echo "vLLM Server Launcher"
echo "=========================================="
echo "MODEL_ID:               $MODEL_ID"
echo "HOST:                   $HOST"
echo "PORT:                   $PORT"
echo "DTYPE:                  $DTYPE"
echo "GPU_MEMORY_UTILIZATION: $GPU_MEMORY_UTILIZATION"
echo "MAX_MODEL_LEN:          $MAX_MODEL_LEN"
echo "TENSOR_PARALLEL_SIZE:   $TENSOR_PARALLEL_SIZE"
echo "=========================================="

# Check if vllm is installed
if ! python -c "import vllm" 2>/dev/null; then
    echo ""
    echo "ERROR: vLLM is not installed."
    echo ""
    echo "To install vLLM, run:"
    echo "  pip install vllm>=0.4.0"
    echo ""
    echo "Note: vLLM requires specific CUDA versions. See:"
    echo "  https://docs.vllm.ai/en/latest/getting_started/installation.html"
    exit 1
fi

cd "$(dirname "$0")/../../.."

# Activate venv if present
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo ""
echo "Starting vLLM server..."
echo "API endpoint will be available at: http://$HOST:$PORT/v1"
echo ""

# Launch vLLM OpenAI-compatible server
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --host "$HOST" \
    --port "$PORT" \
    --dtype "$DTYPE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --trust-remote-code

