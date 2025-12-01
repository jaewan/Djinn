#!/bin/bash
# Launch PyTorch RPC baseline server for Experiment 5.1
# This script sets up the necessary environment variables and launches the RPC server

set -e

# Defaults
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
DEVICE="${DEVICE:-cuda:0}"
DTYPE="${DTYPE:-float16}"
WORKER_NAME="${WORKER_NAME:-rpc_server}"
CONFIG="${CONFIG:-Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml}"
WORKLOADS="${WORKLOADS:-}"
INIT_TIMEOUT="${INIT_TIMEOUT:-0}"

# RPC-specific: server has rank=0, client has rank=1
export MASTER_ADDR
export MASTER_PORT
export RANK=0
export WORLD_SIZE=2

echo "=========================================="
echo "PyTorch RPC Baseline Server"
echo "=========================================="
echo "MASTER_ADDR:    $MASTER_ADDR"
echo "MASTER_PORT:    $MASTER_PORT"
echo "DEVICE:         $DEVICE"
echo "DTYPE:          $DTYPE"
echo "WORKER_NAME:    $WORKER_NAME"
echo "CONFIG:         $CONFIG"
echo "RANK:           $RANK"
echo "WORLD_SIZE:     $WORLD_SIZE"
echo "=========================================="

cd "$(dirname "$0")/../../.."

# Activate venv if present
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

# Launch server
python Evaluation/exp5_1_overhead/scripts/rpc_server.py \
    --config "$CONFIG" \
    --device "$DEVICE" \
    --dtype "$DTYPE" \
    --worker-name "$WORKER_NAME" \
    --init-timeout "$INIT_TIMEOUT" \
    $WORKLOADS

