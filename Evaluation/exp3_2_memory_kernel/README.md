# Experiment 3.2 – Memory Kernel Deep Dive

Demonstrates VMU’s OS-level guarantees with Djinn-specific workloads (concurrent
stateful sessions). Mirrors §6.3 in `docs/EvaluationPlan.md`.

## Goals
1. Show zero external fragmentation under allocate/use/free churn.
2. Prove sub-20 µs allocation latency vs. cudaMalloc / PyTorch caching allocator.
3. Verify session isolation: each conversation’s KV cache is fenced.

## Structure
- `configs/session_stress.yaml` – parameters for session workloads and allocator
  backends.
- `scripts/run_memory_stress.py` – orchestrates synthetic sessions, collects
  fragmentation/latency stats.
- `results/` – raw metrics (JSON/CSV), one per allocator/run.

## Usage
```bash
source .venv/bin/activate

# Synthetic harness (fast dev loop)
python Evaluation/exp3_2_memory_kernel/scripts/run_memory_stress.py \
  --config Evaluation/exp3_2_memory_kernel/configs/session_stress.yaml \
  --allocator vmu --duration 30 \
  --output Evaluation/exp3_2_memory_kernel/results/vmu_synthetic.json

# Real GPU-backed harness (captures VMU metrics + diagnostics)
python Evaluation/exp3_2_memory_kernel/scripts/run_memory_kernel.py \
  --config Evaluation/exp3_2_memory_kernel/configs/session_stress.yaml \
  --allocator vmu --duration 20 \
  --output Evaluation/exp3_2_memory_kernel/results/vmu_real.json
```

You can swap `--allocator` between `vmu` and `torch` for the real runner to
compare Djinn’s segmented VMU against PyTorch’s caching allocator on the same
GPU. The JSON payloads include timeline events, per-session fragmentation stats,
and final VMU diagnostics for plotting.


