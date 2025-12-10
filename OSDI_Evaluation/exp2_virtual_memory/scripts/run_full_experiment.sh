#!/bin/bash
# Full Experiment 2: Memory Virtualization Comparison
# Compares Djinn's block-granularity ring buffer against baselines

set -e

MODEL="meta-llama/Llama-2-13b-hf"
RESULTS_DIR="results/exp2_full"
SCRIPTS_DIR="OSDI_Evaluation/exp2_virtual_memory/scripts"

echo "========================================================================"
echo "Experiment 2: Memory Virtualization (Full Baseline Comparison)"
echo "========================================================================"
echo "Model: $MODEL"
echo "Results directory: $RESULTS_DIR"
echo ""

# Create results directory
mkdir -p $RESULTS_DIR

# Baseline 1: GPU-Only (will OOM if model doesn't fit, but Llama-2-13B should fit on L4)
echo "========================================================================"
echo "Baseline 1: GPU-Only (Native PyTorch)"
echo "========================================================================"
python3 $SCRIPTS_DIR/baseline_gpu_only.py \
    --model $MODEL \
    --output $RESULTS_DIR/gpu_only.json \
    --runs 5 \
    --ttft-enabled \
    2>&1 | tee $RESULTS_DIR/gpu_only.log || echo "GPU-Only baseline failed (expected if OOM)"

# Baseline 2: HuggingFace Accelerate (device_map="auto")
echo ""
echo "========================================================================"
echo "Baseline 2: HuggingFace Accelerate (device_map=auto)"
echo "========================================================================"
python3 $SCRIPTS_DIR/baseline_hf_accelerate.py \
    --model $MODEL \
    --output $RESULTS_DIR/hf_accelerate.json \
    --warmup-runs 2 \
    --measurement-runs 5 \
    2>&1 | tee $RESULTS_DIR/hf_accelerate.log

# Baseline 3: Synchronous Offloading (Binary offload)
echo ""
echo "========================================================================"
echo "Baseline 3: Synchronous CPU Offloading"
echo "========================================================================"
python3 $SCRIPTS_DIR/baseline_synchronous_offload.py \
    --model $MODEL \
    --output $RESULTS_DIR/synchronous_offload.json \
    2>&1 | tee $RESULTS_DIR/synchronous_offload.log

# Djinn: Ring Buffer with Block-Granularity Streaming
echo ""
echo "========================================================================"
echo "Djinn: Block-Granularity Ring Buffer (20GB)"
echo "========================================================================"
python3 $SCRIPTS_DIR/djinn_ring_buffer_measurement.py \
    --model $MODEL \
    --ring-buffer-gb 20.0 \
    --output $RESULTS_DIR/djinn_ring_buffer.json \
    --warmup-runs 2 \
    --measurement-runs 5 \
    2>&1 | tee $RESULTS_DIR/djinn_ring_buffer.log

echo ""
echo "========================================================================"
echo "Experiment 2 Complete!"
echo "========================================================================"
echo "Results saved to: $RESULTS_DIR/"
echo ""
echo "Summary:"
ls -lh $RESULTS_DIR/*.json
echo ""
echo "To generate comparison plots:"
echo "  python3 $SCRIPTS_DIR/plot_comparison.py --results-dir $RESULTS_DIR"

