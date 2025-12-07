#!/bin/bash
#
# Start Djinn Server with Ring Buffer Enabled
#
# This script starts the Djinn server configured for Experiment 2:
# - Ring buffer enabled (20GB capacity)
# - KV swapping disabled (weights own 100% of PCIe bus)
# - Single GPU (L4)
#

set -e

RESULTS_DIR="${1:-/tmp/djinn_server}"
RING_BUFFER_GB="${2:-20}"

mkdir -p "$RESULTS_DIR"

echo "Starting Djinn Server with Ring Buffer..."
echo "  Ring Buffer: ${RING_BUFFER_GB}GB"
echo "  Results: $RESULTS_DIR"
echo ""

# Environment configuration
export GENIE_VMU_RING_BUFFER=true
export GENIE_VMU_RING_BUFFER_GB="$RING_BUFFER_GB"
export DJINN_DISABLE_KV_SWAP=1
export CUDA_VISIBLE_DEVICES=0

# Start server in background
cd /home/jae/Djinn

python3 -m djinn.server.server_main \
    --gpu 0 \
    --port 5556 \
    > "$RESULTS_DIR/server.log" 2>&1 &

SERVER_PID=$!
echo $SERVER_PID > "$RESULTS_DIR/server.pid"

echo "✅ Djinn server started (PID: $SERVER_PID)"
echo "   Port: 5556"
echo "   Logs: $RESULTS_DIR/server.log"
echo ""
echo "Waiting for server to be ready..."
sleep 5

# Check if server is still running
if ps -p $SERVER_PID > /dev/null; then
    echo "✅ Server is running"
else
    echo "❌ Server failed to start. Check logs:"
    tail -20 "$RESULTS_DIR/server.log"
    exit 1
fi
