# Experiment 4.1 – Multi-Tenant Scalability Sweep

Implements the load sweep described in §6.4.1 of `docs/EvaluationPlan.md`.
We vary concurrent user count, enforce Poisson arrivals, and record throughput,
latency, and GPU utilization (or simulator estimates). Results feed Figure 10.

## Layout
- `configs/load_sweep.yaml` – concurrency levels, arrival rates, request mix.
- `scripts/run_load_sweep.py` – asyncio load generator with pluggable drivers
  (Synthetic for local validation, DjinnDriver for real server runs).
- `results/` – JSON dumps per run.
- `notes/` – scratch space for observations.

## Usage
```bash
source .venv/bin/activate
python Evaluation/exp4_1_scalability/scripts/run_load_sweep.py \
  --config Evaluation/exp4_1_scalability/configs/load_sweep.yaml \
  --driver synthetic \
  --duration 20 \
  --output Evaluation/exp4_1_scalability/results/synthetic_smoke.json
```

For real networking runs swap `--driver djinn` and provide `--djinn-endpoint`
(`host:port`). The script will initialize `EnhancedModelManager`, register the
model once, and fan out concurrent `execute_model` calls with QoS hints.

## Djinn Initialization
When `--driver djinn` is selected the harness calls `djinn.init(server_address=...)`
before any load points execute. The data port comes from `--djinn-endpoint` or the
`GENIE_SERVER_ADDRESS` env var. Initialization is performed once so run time still
measures just the workload (`run_point`) latency/throughput.

## Next steps
- Wire `DjinnLoadDriver` to actual coordinator once the remote server is up.
- Add NVML sampling hooks to capture real GPU util curves.
- Integrate QoS tagging so we can reuse this harness for Experiment 4.2.

