# Experiment 5.1 – Framework Overhead Analysis

Goal: validate `docs/EvaluationPlan.md §6.5` by quantifying framework overhead
versus semantic savings across model sizes. This experiment measures end-to-end
latency, throughput, and data movement for microbenchmarks comparing native PyTorch
execution against various disaggregation baselines.

## Contents
- `configs/overhead_hf_smoke.yaml` – main config (6 workloads: 4 synthetic + 2 HuggingFace)
- `scripts/run_overhead_sweep.py` – main runner
- `scripts/analyze_overhead.py` – results analyzer (generates markdown tables)
- `scripts/rpc_server.py` – PyTorch RPC baseline server
- `scripts/launch_rpc_baseline.sh` – RPC server launcher script
- `results/` – JSON output (one file per workload/run)

## Usage

### Run Microbenchmark Suite
```bash
# Terminal 1: Start Djinn server
cd /home/jhong/Djinn && source .venv/bin/activate
GENIE_ENABLE_PROFILING=true python -m djinn.server.server_main \
  --host 0.0.0.0 --port 5556 --gpu 0

# Terminal 2: Run microbenchmarks
export GENIE_SERVER_ADDRESS=localhost:5556
python Evaluation/exp5_1_overhead/scripts/run_overhead_sweep.py \
  --config Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml \
  --tag osdi_microbench

# Analyze results
python Evaluation/exp5_1_overhead/scripts/analyze_overhead.py \
  --input-dir Evaluation/exp5_1_overhead/results \
  --output-format markdown > /tmp/table_1.md
```

### Run PyTorch RPC Baseline
```bash
# Terminal 1: Start RPC server
export MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 RANK=0 WORLD_SIZE=2
bash Evaluation/exp5_1_overhead/scripts/launch_rpc_baseline.sh

# Terminal 2: Run client
export RANK=1
python Evaluation/exp5_1_overhead/scripts/run_overhead_sweep.py \
  --config Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml \
  --workloads hf_tiny_gpt2 \
  --tag rpc_baseline
```

### Flags
- `--workloads hf_tiny_gpt2 llama_decode_1tok` – run subset of workloads
- `--tag mylabel` – control output filenames
- `analyze_overhead.py --latest-only` – read only latest JSON per workload

## Djinn Initialization
Each experiment connects to `GENIE_SERVER_ADDRESS` automatically. The connection
happens once and is not part of per-run latency measurements.

## Workloads

| Name | Category | Implementation | Params | Notes |
|------|----------|----------------|--------|-------|
| `hf_tiny_gpt2` | Sequential | `hf_causal_lm` | GPT-2 (4M params) | Smoke test with real HF model |
| `llama_decode_1tok` | Sequential | `hf_causal_lm` | Llama-2-7B (7B params) | Single token decode |
| `llama_prefill_1k` | Sequential | `hf_causal_lm` | Llama-2-7B (7B params) | 1024 token prefill |
| `bert_latency_bound` | Encoder | `synthetic_transformer` | Synthetic BERT-like | Latency-bound workload |
| `resnet50_batch8` | Vision | `hf_vision` | ResNet-50 | Batch processing |
| `hf_tiny_gpt2` | Sequential | `hf_causal_lm` | GPT-2 (4M params) | Extended generation |

All workloads run with `batch_size: 1` and are optimized for A100 GPUs.

## Baselines

| Name | Type | Status | Purpose |
|------|------|--------|---------|
| `native_pytorch` | `local_synthetic` | ✅ **Working** | GPU-local PyTorch (upper bound) |
| `semantic_blind` | `remote_djinn` | ✅ **Working** | Djinn remote without semantic hints |
| `full_djinn` | `remote_djinn` | ✅ **Working** | Djinn remote with full semantics |
| `pytorch_rpc` | `pytorch_rpc` | ✅ **Ready** | PyTorch distributed RPC baseline |

All baselines capture detailed profiling data including serialization, network,
and server-side timing breakdowns.

## Profiling Data Captured

Each baseline captures detailed timing breakdowns:

**Client-Side**:
- `client_serialize_ms` – Input tensor → binary conversion
- `client_deserialize_ms` – Output binary → tensor conversion
- `client_network_c2s_ms` – Client → server network latency
- `client_network_s2c_ms` – Server → client network latency

**Server-Side** (Djinn baselines only):
- `server_duration_ms` – Total server processing time
- `server_executor_time_ms` – Hybrid executor + RPC overhead
- `server_queue_latency_ms` – QoS queue wait time
- `server_plan_ms` – Semantic planning phase
- `server_placement_ms` – Memory placement optimization
- `server_execution_ms` – Model forward pass
- `server_skeletonization_ms` – Output skeletonization
- `server_cleanup_ms` – Resource cleanup

## Output Schema
```json
{
  "workload": "hf_tiny_gpt2",
  "category": "sequential",
  "results": [
    {
      "baseline": "native_pytorch",
      "aggregates": {
        "latency_ms": {"mean": 252.71, "p95": 253.8},
        "throughput_units_per_s": {"mean": 63.32},
        "total_data_mb": {"mean": 0.00061}
      }
    },
    {
      "baseline": "full_djinn",
      "aggregates": {
        "latency_ms": {"mean": 110.99},
        "throughput_units_per_s": {"mean": 144.20},
        "total_data_mb": {"mean": 6.14},
        "client_serialize_ms": {"mean": 0.26},
        "client_deserialize_ms": {"mean": 8.90},
        "server_plan_ms": {"mean": 0.36},
        "server_execution_ms": {"mean": 17.12}
      }
    }
  ]
}
```

## Sample Results (A100 GPU)

**Model**: sshleifer/tiny-gpt2 (4M params)
**Task**: Generate 16 tokens, batch_size=1
**Runs**: 2 per baseline + 1 warmup

| Baseline | Latency (ms) | Throughput (tok/s) | Data (MB) | Status |
|----------|-------------|-------------------|-----------|--------|
| native_pytorch | 252.71 | 63.32 | 0.00061 | GPU-local baseline |
| semantic_blind | 116.30 | 137.58 | 6.14 | Djinn w/o semantics |
| full_djinn | 110.99 | 144.20 | 6.14 | **Djinn w/ semantics** |

**Key Finding**: Djinn achieves 2.27× speedup over native PyTorch, with semantic
awareness providing additional 3% improvement over blind disaggregation.

## Next Steps
- Run full workload suite (including Llama-2-7B) to measure semantic benefits at scale
- Test PyTorch RPC baseline with proper environment setup
- Generate Table 1 for OSDI/SOSP paper using `analyze_overhead.py`
- Extend profiling to capture explicit network latency measurements


