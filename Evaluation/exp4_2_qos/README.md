# Experiment 4.2 – QoS Guarantees Under Contention

Implements §6.4.2 from `docs/EvaluationPlan.md`. We ingest a workload mix
(Realtime, Interactive, Batch), drive arrivals using Poisson processes, and
compare Djinn's QoS scheduler vs. FCFS. Outputs per-class latency percentiles
and SLA violation rates for Figure 11.

## Layout
- `configs/qos_mix.yaml` – arrival mix, targets, concurrency budget.
- `scripts/run_qos_study.py` – harness with two drivers:
  - `synthetic` (default) – analytic model for quick validation.
  - `djinn` (stub) – future wiring to actual Djinn coordinator.
- `results/` – JSON dumps per policy (`fcfs`, `qos`).

## Usage
```bash
source .venv/bin/activate
python Evaluation/exp4_2_qos/scripts/run_qos_study.py \
  --config Evaluation/exp4_2_qos/configs/qos_mix.yaml \
  --driver synthetic \
  --duration 60 \
  --output Evaluation/exp4_2_qos/results/synthetic_qos.json
```

Swap `--policy` between `fcfs` and `qos` to generate both curves. When the Djinn
driver is wired, pass `--driver djinn --djinn-endpoint host:port` to hit the
real server via `EnhancedModelManager`.

## Next steps
- Integrate Djinn driver once QoS scheduler is live end-to-end.
- Capture NVML metrics to correlate latency guarantees with GPU util.
- Extend harness to record queue depth traces for the paper appendix.

