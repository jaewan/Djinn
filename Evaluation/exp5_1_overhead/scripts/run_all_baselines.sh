#!/bin/bash
#
# Run all 4 core baselines on single machine
# Requires: Djinn server and PyTorch RPC server running
#
# Usage:
#   bash run_all_baselines.sh [config] [workloads]
#
# Example:
#   bash run_all_baselines.sh Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml hf_tiny_gpt2
#
# This script runs 4 baselines for fair OSDI/SOSP comparison:
#   - native_pytorch (local baseline)
#   - pytorch_rpc (canonical RPC baseline)
#   - semantic_blind (Djinn without semantics)
#   - full_djinn (Djinn with full semantics)
#
# Note: vLLM baseline requires separate manual startup and is not included here

set -e

CONFIG="${1:-Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml}"
WORKLOADS="${2:-hf_tiny_gpt2}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$REPO_ROOT"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      Running 4 Core Baselines - Single Machine Setup         ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Config: $CONFIG"
echo "Workloads: $WORKLOADS"
echo ""
echo "Prerequisites:"
echo "  Terminal 1: Djinn server (port 5556)"
echo "  Terminal 2: PyTorch RPC server (port 29500, RANK=0)"
echo "  Terminal 3: This script (client, RANK=1)"
echo ""

# Check if required servers are running
echo "Checking server availability..."

if ! nc -z 127.0.0.1 5556 2>/dev/null; then
    echo ""
    echo "❌ Djinn server not running on localhost:5556"
    echo ""
    echo "Start it with:"
    echo "  export GENIE_VMU_SESSION_ARENA_MB=128"
    echo "  export GENIE_ENABLE_PROFILING=true"
    echo "  python -m djinn.server.server_main --host 127.0.0.1 --port 5556 --gpu 0"
    echo ""
    exit 1
fi
echo "✅ Djinn server ready"

if ! nc -z 127.0.0.1 29500 2>/dev/null; then
    echo ""
    echo "❌ PyTorch RPC server not running on localhost:29500"
    echo ""
    echo "Start it with:"
    echo "  export MASTER_ADDR=127.0.0.1"
    echo "  export MASTER_PORT=29500"
    echo "  export RANK=0"
    echo "  export WORLD_SIZE=2"
    echo "  python Evaluation/exp5_1_overhead/scripts/rpc_server.py \\"
    echo "    --config $CONFIG --workloads $WORKLOADS --device cuda:0"
    echo ""
    exit 1
fi
echo "✅ PyTorch RPC server ready"

# Run the test
echo ""
echo "════════════════════════════════════════════════════════════"
echo "Running evaluation with 4 baselines..."
echo "════════════════════════════════════════════════════════════"
echo ""

export GENIE_SERVER_ADDRESS=localhost:5556
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export RANK=1  # Client rank
export WORLD_SIZE=2
export PYTHONPATH=$REPO_ROOT:$PYTHONPATH

python "$SCRIPT_DIR/run_overhead_sweep.py" \
    --config "$CONFIG" \
    --workloads "$WORKLOADS" \
    --tag all_baselines_single_machine

echo ""
echo "✅ All baselines complete!"
echo ""
echo "Analyze results with:"
echo "  python Evaluation/exp5_1_overhead/scripts/analyze_overhead.py \\"
echo "    --results-dir Evaluation/exp5_1_overhead/results \\"
echo "    --workloads $WORKLOADS --latest-only --format markdown"

