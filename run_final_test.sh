#!/bin/bash

set -e

cd /home/ubuntu/Djinn

# Activate venv in this shell
source .venv/bin/activate

# Set environment
export GENIE_VMU_SESSION_ARENA_MB=64

# Kill old servers
pkill -9 -f "server_main" 2>/dev/null || true
sleep 3

# Start new server
python3 -m djinn.server.server_main --port 5556 --gpu 0 --enable-semantic-scheduler --idle-threshold-seconds 1.0 --host-swap-pool-gb 32 > /tmp/djinn_final_server.log 2>&1 &

sleep 25
echo "✅ Server started"

# Run experiment
timeout 600 python3 OSDI_Evaluation/exp1_semantic_scheduler/scripts/run_experiment.py \
  --config OSDI_Evaluation/exp1_semantic_scheduler/configs/agent_scaling_osdi_science.yaml \
  --model-id meta-llama/Llama-2-13b-hf \
  --djinn-server localhost:5556 \
  --output-dir OSDI_Evaluation/exp1_semantic_scheduler/results \
  2>&1 | tee /tmp/final_osdi_test_v3.log

echo ""
echo "════════════════════════════════════════════"
echo "🔍 FINAL RESULTS:"
echo "════════════════════════════════════════════"
tail -20 /tmp/final_osdi_test_v3.log | grep -E "N=|swaps|restores"

