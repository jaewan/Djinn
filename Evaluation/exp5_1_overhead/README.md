# Experiment 5.1 – Framework Overhead Analysis

Goal: validate `docs/EvaluationPlan.md §6.5` by quantifying framework overhead
versus semantic savings across model sizes.  The current patch wires up the
automation so we can iterate on a dev-sized L4 GPU using synthetic workloads.
Once the Djinn remote baselines are ready the same harness can run with real
models by editing the YAML (no code changes).

## Contents
- `configs/overhead_smoke.yaml` – default config (3 synthetic workloads, 3 baselines)
- `scripts/run_overhead_sweep.py` – main runner
- `results/` – JSON output (one file per workload/run)
- `notes/` – stash for derived plots/tables

## Usage
```bash
source .venv/bin/activate
python Evaluation/exp5_1_overhead/scripts/run_overhead_sweep.py \
  --config Evaluation/exp5_1_overhead/configs/overhead_smoke.yaml \
  --tag smoke

# After runs finish, summarize:
python Evaluation/exp5_1_overhead/scripts/analyze_overhead.py \
  --results-dir Evaluation/exp5_1_overhead/results \
  --format markdown
```

Flags:
- `--workloads llama_like` to run a subset
- `--output-dir /tmp/exp5_1` to override the default
- `--tag mylabel` to control filenames
- `analyze_overhead.py --latest-only` to read only the latest JSON per workload

## Djinn Initialization
Each experiment obtains `djinn.init(server_address=...)` automatically before any measurement; the `djinn_server_address` field in `configs/overhead_smoke.yaml` (or the env var `GENIE_SERVER_ADDRESS`) tells the harness which data port to connect to. The init happens once and is not part of the per-run latency stats, so measurements only include the workload work.

Each run writes `results/<workload>/<tag>_<timestamp>.json` containing all
baseline metrics plus derived speedup/overhead fields used in the paper figures.

## Smoke-Test Workloads

| Name | Category | Implementation | Notes |
|------|----------|----------------|-------|
| `llama_like` | Sequential/stateful | `synthetic_transformer` | Approximates 7B decode |
| `bert_like` | Encoder | `synthetic_transformer` | Shorter sequence to mimic BERT |
| `vit_like` | Vision/parallel | `synthetic_cnn` | Conv stack for 224×224 inputs |
| `hf_tiny_gpt2` | Sequential/stateful | `hf_causal_lm` | Real HuggingFace decode (see `overhead_hf_smoke.yaml`) |

The synthetic models keep VRAM under 4 GB so the harness runs comfortably on the
L4 dev box.  Swap in real HuggingFace identifiers later by adding new workloads
that set `implementation: huggingface_*` (runner TBD).

## Baselines

| Name | Type | Purpose |
|------|------|---------|
| `native_pytorch` | `local_synthetic` | Upper bound / pure PyTorch |
| `semantic_blind` | `scaling` (relative to native) | Placeholder for Djinn semantics disabled |
| `full_djinn` | `scaling` (relative to native) | Placeholder for Djinn full semantics |

`semantic_blind` and `full_djinn` currently scale the native measurements using
factors from the YAML to unblock analysis scripts.  Replace them with real
baseline definitions (e.g., `type: djinn_remote`) as soon as the server harness
lands.

## Output Schema
```json
{
  "experiment": {...},
  "workload": {
    "workload": "llama_like",
    "category": "sequential",
    "timestamp": "...",
    "results": [
      {
        "baseline": "native_pytorch",
        "aggregates": {
          "latency_ms": {"mean": 42.1, "p95": 44.8, ...},
          "total_data_mb": {"mean": 1.2, ...},
          "throughput_units_per_s": {"mean": 305.3, ...}
        },
        "derived": {
          "latency_overhead_pct_vs_native_pytorch": 0.0
        }
      },
      {
        "baseline": "full_djinn",
        "derived": {
          "speedup_vs_semantic_blind": 3.4,
          "data_savings_pct_vs_semantic_blind": 98.7,
          "semantic_efficiency_ratio": 140.0
        }
      }
    ]
  }
}
```

## Next Steps
- Swap synthetic workloads for real HuggingFace/Djinn runners.
- Use `configs/overhead_hf_smoke.yaml` with `run_overhead_sweep.py` to sanity-check
  the HuggingFace-backed workload (`implementation: hf_causal_lm`).
- Use the analyzer to convert JSON into Figure 12 (overhead vs model size).
- Integrate GPU utilization sampling via `Evaluation.common.gpu`.


