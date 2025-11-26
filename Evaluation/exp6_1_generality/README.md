# Experiment 6.1 – Cross-Workload Generality

Implements the week-5 generality study from `docs/EvaluationPlan.md §6.6`.  The
goal is to demonstrate that Djinn’s semantic benefits extend beyond LLM decode
by exercising sequential, parallel, and hybrid workloads and summarizing
speedup/data-savings per category.

## Files
- `configs/generality_smoke.yaml` – grouped workloads + baseline definitions
- `scripts/run_generality_suite.py` – orchestrates runs + aggregation
- `results/` – JSON dumps with per-group data + summary table

## Running the Smoke Test
```bash
source .venv/bin/activate
python Evaluation/exp6_1_generality/scripts/run_generality_suite.py \
  --config Evaluation/exp6_1_generality/configs/generality_smoke.yaml \
  --tag smoke

# Summarize per-group metrics:
python Evaluation/exp6_1_generality/scripts/analyze_generality.py \
  --result-file Evaluation/exp6_1_generality/results/smoke.json \
  --format markdown
```

Use `--groups sequential hybrid` to run a subset.  The script prints per-workload
latency/data plus per-group semantic efficiency statistics.

## Current Workloads

| Group | Workload | Implementation | Notes |
|-------|----------|----------------|-------|
| Sequential | `seq_llama_like`, `seq_audio_like` | `synthetic_transformer` | Token/state heavy |
| Parallel | `par_resnet_like`, `par_vit_like` | `synthetic_cnn` | Vision-style throughput |
| Hybrid | `hybrid_clip_like` | `synthetic_hybrid` | CNN + transformer mix |

Like Experiment 5.1, the smoke configuration keeps VRAM demand low so we can
iterate on the dev box.  Swap `implementation` to real HuggingFace/Djinn runners
once they are ready.

## Output

Top-level JSON structure:
```json
{
  "groups": [
    {
      "group": "sequential",
      "workloads": [...],
      "description": "LLM/ASR style"
    }
  ],
  "summaries": [
    {
      "group": "sequential",
      "metrics": {
        "speedup_vs_blind": {"mean": 3.4, "p95": 3.6},
        "data_savings_pct_vs_blind": {"mean": 98.9},
        "semantic_efficiency_ratio": {"mean": 120.0}
      }
    }
  ]
}
```

These summaries feed Figure 13/Table 5 in the evaluation plan.

## Next Steps
- Replace scaling placeholders with real Djinn baselines.
- Add metric for per-group success rate (how often semantic planner selects
  correct phase).
- Use analyzer output to drive CSV/plotting pipeline for Figure 13/Table 5.

## Djinn Initialization
Workloads now call `djinn.init(server_address=...)` before executing; the config key `djinn_server_address` (or `GENIE_SERVER_ADDRESS`) determines the target data port and the init happens once, outside of the timed workloads, so the reported latency/throughput numbers are not contaminated by initialization time.


