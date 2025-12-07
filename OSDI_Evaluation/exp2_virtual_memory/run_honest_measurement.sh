#!/bin/bash
#
# Honest Measurement: TTFT vs Decode vs E2E
#
# This script measures what ACTUALLY happens with the ring buffer:
# 1. TTFT: Time to first token (prefill only)
# 2. Decode latency per token (what's the cost of each autoregressive token?)
# 3. E2E: Full 50-token generation
#
# Key insight: Ring buffer must re-stream non-resident weights for EACH token during decode
#

set -e

cd /home/jae/Djinn

export CUDA_VISIBLE_DEVICES=0

MODEL="meta-llama/Llama-2-13b-hf"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="OSDI_Evaluation/exp2_virtual_memory/results/honest_measurement_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║        HONEST MEASUREMENT: TTFT vs Decode vs E2E                         ║"
echo "║                                                                          ║"
echo "║  Measuring REAL latencies without simulation or tricks                   ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Time: $(date)"
echo "Model: $MODEL (26GB on 24GB GPU)"
echo "Results: $RESULTS_DIR"
echo ""

python3 OSDI_Evaluation/exp2_virtual_memory/scripts/djinn_client_honest_measurement.py \
    --model "$MODEL" \
    --output "$RESULTS_DIR/honest_measurements.json" \
    2>&1 | tee "$RESULTS_DIR/measurement.log"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                  ✅ HONEST MEASUREMENT COMPLETE                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Files generated:"
ls -lh "$RESULTS_DIR"/*.json "$RESULTS_DIR"/*.log 2>/dev/null | awk '{print "  " $9, "(" $5 ")"}'
echo ""

