# Experiment 2.1 – LLM Decode with KV Cache

Hero experiment for `docs/EvaluationPlan.md §6.2`.  Objective: prove Djinn’s
semantic awareness transforms disaggregated execution during LLM decoding by
demonstrating 30–57× lower per-token latency, 10³–10⁵× less data transfer, and
dramatically higher GPU utilization relative to semantic-blind variants.

## Workload
- **Model:** default `meta-llama/Llama-2-7b-hf` (override via CLI flag)
- **Prompt:** 72-token prompt (see `configs/prompt.txt`)
- **Output:** 50 new tokens (`max_new_tokens=50`)
- **Batch size:** 1 (repeatable)
- **Transport:** Client CPU → Djinn server (A100-80GB) over 25 Gbps link

## Baselines
| Name | Location | Purpose | Config reference |
|------|----------|---------|------------------|
| `native_pytorch` | local GPU | Upper bound | `configs/baselines.yaml` |
| `semantic_blind` | Djinn, semantics disabled | Quantify worst-case | `configs/baselines.yaml` |
| `partially_aware` | Djinn, caching w/o KV semantics | Incremental | `configs/baselines.yaml` |
| `full_djinn` | Djinn, all semantics | Target system | `configs/baselines.yaml` |

Baseline toggles are documented in `configs/baselines.yaml`.  The local PyTorch
runner is implemented in `scripts/run_local_baseline.py`; Djinn variants will
reuse the shared harness in a follow-up patch once QoS + semantic knobs are wired
through config.

## Metrics
- Per-token latency (ms) and total decode time
- Data transferred host↔device per token (computed via payload logs)
- GPU SM utilization (%), memory BW utilization (%)
- Queue/compute/transfer breakdown (via telemetry timestamps)
- Energy / power samples (optional via `nvidia-smi --query-gpu=power.draw`)

## Data Capture Plan
1. Run 30 iterations per baseline (discard first 5 for warm-up).
2. Save raw run JSON under `results/<baseline>/<timestamp>.json` with schema:
   ```json
   {
     "baseline": "native_pytorch",
     "model_id": "meta-llama/Llama-2-7b-hf",
     "prompt_tokens": 72,
     "new_tokens": 50,
     "batch_size": 1,
     "runs": [
       {
         "run_id": 1,
         "total_ms": 812.4,
         "per_token_ms": 16.2,
         "tokens_generated": 50,
         "gpu_util_pct": 84.3,
         "memory_bw_gbps": 1.2,
         "host_to_device_mb": 0.8,
         "device_to_host_mb": 0.1
       }
     ],
     "aggregates": {
       "mean_total_ms": 812.4,
       "p95_total_ms": 830.1,
       "mean_per_token_ms": 16.2
     }
   }
   ```
3. `scripts/analyze_results.py` (placeholder) will merge per-baseline JSON into
   CSV for plotting.

## Instructions
1. Ensure Djinn server is running for remote baselines; for PyTorch baseline a
   single A100 is sufficient.
2. Install dependencies:
   ```bash
   pip install -r requirements-dev.txt
   pip install transformers accelerate datasets pynvml
   ```
3. Run local baseline as smoke test:
   ```bash
   python Evaluation/exp2_1_llm_decode/scripts/run_local_baseline.py \
     --model-id meta-llama/Llama-2-7b-hf \
     --prompt-file Evaluation/exp2_1_llm_decode/configs/prompt.txt \
     --output-dir Evaluation/exp2_1_llm_decode/results/native_pytorch \
     --runs 3
   ```
4. Validate JSON output and aggregate stats.
5. Integrate telemetry hooks for Djinn baselines (pending).

## Remote Djinn Baselines
When the server is available, run Djinn backends by passing `--backend djinn`.
The device string switches to `privateuseone:<N>` so execution happens on the remote
accelerator; `GENIE_SERVER_ADDRESS` or `--djinn-server` controls the target port.

```
source .venv/bin/activate
export DJINN_ENABLE_PRIVATEUSE1_DEVICE=1
python Evaluation/exp2_1_llm_decode/scripts/run_local_baseline.py \
  --backend djinn \
  --djinn-server localhost:5556 \
  --model-id sshleifer/tiny-gpt2 \
  --prompt-file Evaluation/exp2_1_llm_decode/configs/prompt.txt \
  --runs 1 \
  --warmup-runs 1 \
  --tag djinn_remote_smoke \
  --output-dir Evaluation/exp2_1_llm_decode/results/djinn_remote
```

Tail the Djinn server logs to ensure transport/connectivity and compare the
`backend`, `djinn_server`, and latency fields in the resulting JSON.

## Open Tasks
- [ ] Add Djinn remote executor harness with semantic toggles.
- [ ] Stream GPU utilization samples via NVML instead of relying on `nvidia-smi`.
- [ ] Build `analyze_results.py` to emit Figures 6a–6c.
- [ ] Automate prompt + seed selection to match paper artifact.

