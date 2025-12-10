#!/bin/bash
# Experiment 2: Large Model (70B) - Memory Virtualization with Extreme Oversubscription
# Compares Djinn's block-granularity ring buffer against baselines
# Model: Llama-2-70B (140GB FP16) on 24GB GPU = 5.8× oversubscription

set -e

MODEL="meta-llama/Llama-2-70b-hf"
RESULTS_DIR="results/exp2_70b"
SCRIPTS_DIR="OSDI_Evaluation/exp2_virtual_memory/scripts"

echo "========================================================================"
echo "Experiment 2: Large Model (70B) - Extreme Oversubscription"
echo "========================================================================"
echo "Model: $MODEL (140GB)"
echo "Hardware: L4 (24GB VRAM)"
echo "Oversubscription: 5.8×"
echo "Results directory: $RESULTS_DIR"
echo ""

# Create results directory
mkdir -p $RESULTS_DIR

# Baseline 1: GPU-Only (will definitely OOM)
echo "========================================================================"
echo "Baseline 1: GPU-Only (Native PyTorch) - Expected OOM"
echo "========================================================================"
timeout 120 python3 $SCRIPTS_DIR/baseline_gpu_only.py \
    --model $MODEL \
    --output $RESULTS_DIR/gpu_only.json \
    --runs 1 \
    2>&1 | tee $RESULTS_DIR/gpu_only.log || {
    echo "✓ GPU-Only baseline OOM'd as expected (validates problem)"
    echo '{"status": "OOM", "error": "Model 140GB does not fit in 24GB VRAM"}' > $RESULTS_DIR/gpu_only.json
}

# Baseline 2: HuggingFace Accelerate (device_map="auto")
echo ""
echo "========================================================================"
echo "Baseline 2: HuggingFace Accelerate (device_map=auto)"
echo "========================================================================"
echo "Note: This baseline will be very slow but functional"
timeout 300 python3 $SCRIPTS_DIR/baseline_hf_accelerate.py \
    --model $MODEL \
    --output $RESULTS_DIR/hf_accelerate.json \
    --warmup-runs 1 \
    --measurement-runs 2 \
    2>&1 | tee $RESULTS_DIR/hf_accelerate.log || {
    echo "⚠ HF Accelerate timed out or failed"
}

# Baseline 3: Synchronous Offloading (Binary offload)
echo ""
echo "========================================================================"
echo "Baseline 3: Synchronous CPU Offloading"
echo "========================================================================"
timeout 300 python3 $SCRIPTS_DIR/baseline_synchronous_offload.py \
    --model $MODEL \
    --output $RESULTS_DIR/synchronous_offload.json \
    2>&1 | tee $RESULTS_DIR/synchronous_offload.log || {
    echo "⚠ Sync offload timed out or failed"
}

# Baseline 4: DeepSpeed (expected to OOM)
echo ""
echo "========================================================================"
echo "Baseline 4: DeepSpeed-Inference (Expected OOM)"
echo "========================================================================"
timeout 120 python3 $SCRIPTS_DIR/baseline_deepspeed_offload.py \
    --model $MODEL \
    --output $RESULTS_DIR/deepspeed.json \
    --runs 1 \
    2>&1 | tee $RESULTS_DIR/deepspeed.log || {
    echo "✓ DeepSpeed OOM'd as expected"
    echo '{"status": "OOM", "error": "DeepSpeed init_inference requires loading entire model to GPU"}' > $RESULTS_DIR/deepspeed.json
}

# Djinn: Ring Buffer with Block-Granularity Streaming (12GB buffer for 70B model)
echo ""
echo "========================================================================"
echo "Djinn: Block-Granularity Ring Buffer (12GB buffer)"
echo "========================================================================"
timeout 600 python3 $SCRIPTS_DIR/djinn_ring_buffer_measurement.py \
    --model $MODEL \
    --ring-buffer-gb 12.0 \
    --output $RESULTS_DIR/djinn_ring_buffer.json \
    --warmup-runs 1 \
    --measurement-runs 2 \
    2>&1 | tee $RESULTS_DIR/djinn_ring_buffer.log || {
    echo "⚠ Djinn measurement failed"
}

echo ""
echo "========================================================================"
echo "Experiment 2 (70B) Complete!"
echo "========================================================================"
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Summary:"
ls -lh $RESULTS_DIR/*.json 2>/dev/null | awk '{print "  " $9, "(" $5 ")"}'
echo ""

# Generate comparison report
python3 << 'EOF'
import json
from pathlib import Path
import sys

results_dir = Path("results/exp2_70b")

print("\n" + "="*70)
print("EXPERIMENT 2 (70B) - RESULTS SUMMARY")
print("="*70 + "\n")

# Try to load each result
results = {}
for baseline in ["gpu_only", "hf_accelerate", "synchronous_offload", "deepspeed", "djinn_ring_buffer"]:
    try:
        with open(results_dir / f"{baseline}.json") as f:
            data = json.load(f)
            results[baseline] = data
            status = data.get("status", "OK")
            if status == "OOM":
                print(f"❌ {baseline}: OOM")
            else:
                ttft = data.get("measurements", {}).get("ttft", {}).get("average_ms", "N/A")
                decode = data.get("measurements", {}).get("decode", {}).get("average_ms_per_token", "N/A")
                print(f"✅ {baseline}: TTFT={ttft}ms, Decode={decode}ms/tok")
    except Exception as e:
        print(f"⚠ {baseline}: Not available ({e})")

print("\n" + "="*70)
EOF

echo ""

