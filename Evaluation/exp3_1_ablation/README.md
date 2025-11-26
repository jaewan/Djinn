# Experiment 3.1 – Component Ablation Study

Validates how each semantic component contributes to LLM throughput/latency
(`docs/EvaluationPlan.md §6.3`). We start from full Djinn, disable one feature
at a time, and run two workloads (LLaMA-7B decode, ResNet-50 batch). This
workspace holds configs and scripts to automate those runs.

## Config Layout
- `configs/ablation_matrix.yaml` – authoritative list of configurations, toggles,
  workloads, and metrics to collect.
- `configs/workloads.yaml` – per-workload parameters (token counts, batch size,
  input assets).

## Scripts
- `scripts/render_matrix.py` – prints the expanded run matrix, ensuring configs
  parse correctly (used as a smoke test on the dev server before launching
  heavy runs).
- Future: `scripts/run_ablation.py` to invoke Djinn harness automatically once
  remote execution is wired up.

## Usage
```bash
source .venv/bin/activate
python Evaluation/exp3_1_ablation/scripts/render_matrix.py \
  --matrix Evaluation/exp3_1_ablation/configs/ablation_matrix.yaml \
  --workloads Evaluation/exp3_1_ablation/configs/workloads.yaml
```

This produces a JSON summary of all (workload × configuration) combinations,
confirming the toggles we implemented. Once the Djinn runner is ready, the same
YAML will drive actual experiments.

## Next Steps
1. Integrate with the remote execution harness (reuse Experiment 2.1 runner).
2. Add metrics ingestion + plotting for Figures 8a–8b.
3. Run smoke tests on the dev server with smaller models before moving to the
   expensive evaluation cluster.

