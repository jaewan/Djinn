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

## Ray Keep-Alive Baseline
To reproduce the “persistent actor” comparison, launch Ray locally (or point to
an existing cluster) and run:

```bash
python Evaluation/exp2_1_llm_decode/scripts/run_ray_keepalive_agents.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --prompt-file Evaluation/exp2_1_llm_decode/configs/prompt.txt \
  --agent-counts 1 2 4 8 \
  --iterations 1 \
  --sleep-seconds 10 \
  --output-dir Evaluation/exp2_1_llm_decode/results/ray_keepalive
```

Each agent receives a dedicated Ray actor that pins the full model + KV cache on
GPU memory for the duration of the experiment.  The output JSON matches the
structure used by other hero baselines, enabling direct comparison of per-stage
latency across concurrency levels.

## Ray Serverless Baseline
For the stateless/serverless comparison (load/unload per step), run:

```bash
python Evaluation/exp2_1_llm_decode/scripts/run_ray_serverless_agents.py \
  --model-id meta-llama/Llama-2-7b-hf \
  --prompt-file Evaluation/exp2_1_llm_decode/configs/prompt.txt \
  --agent-counts 1 2 4 8 \
  --iterations 1 \
  --sleep-seconds 10 \
  --output-dir Evaluation/exp2_1_llm_decode/results/ray_serverless
```

Each `Reason`/`Reflect` step executes as a fresh Ray task that reloads the
weights, recomputes the prompt, and discards state afterwards, capturing the
cold-start costs inherent to this design.

## Experiment 2.2: Agent Scaling (Hero Experiment)

This is the main hero figure for the paper. We compare three baselines on the
"Parking Lot" problem: running N concurrent agents that do Reason → Act (sleep) → Reflect.

### Baselines

1. **Ray Keep-Alive** (`run_ray_keepalive_agents.py`):
   - Each agent holds model + KV cache exclusively
   - Expected: Flat latency, but OOMs at low N (~2-3 agents)

2. **Ray Serverless** (`run_ray_serverless_agents.py`):
   - Each step reloads model + KV cache
   - Expected: Scales to N=32, but latency is high (~5-8s per step)

3. **Djinn** (`run_djinn_agents.py`):
   - Shared weights (Text Segment), per-session KV cache (Data Segment)
   - **True incremental decode**: REFLECT phase sends only new tokens, reusing KV cache
   - Expected: Scales to N=32 with low latency (~150ms per step)
   - Expected: REFLECT latency ~5-20x faster than REASON (proves KV reuse)

### Quick Start (Smoke Test)

**Option 1: Sequential Runner (Recommended)**

Run all three baselines sequentially with automatic server management:

```bash
# Ensure Ray cluster is running
ray start --head --num-gpus=1 --include-dashboard=false

# Run all baselines sequentially (server starts/stops automatically)
python Evaluation/exp2_1_llm_decode/scripts/run_all_baselines_sequential.py \
  --agent-counts 1 2 \
  --iterations 1 \
  --new-tokens 10 \
  --sleep-seconds 1.0 \
  --model-id sshleifer/tiny-gpt2 \
  --output-dir /tmp/test_agent_scaling
```

**Option 2: Manual Execution**

```bash
# Terminal 1: Start Djinn server
python -m djinn.server.server_main --gpu 0 --port 5556

# Terminal 2: Run smoke test (1-2 agents only)
# Ray Keep-Alive
python Evaluation/exp2_1_llm_decode/scripts/run_ray_keepalive_agents.py \
  --agent-counts 1 2 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 1 \
  --output-dir Evaluation/exp2_1_llm_decode/results/ray_keepalive

# Ray Serverless
python Evaluation/exp2_1_llm_decode/scripts/run_ray_serverless_agents.py \
  --agent-counts 1 2 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 1 \
  --output-dir Evaluation/exp2_1_llm_decode/results/ray_serverless

# Djinn (requires Djinn server running)
python Evaluation/exp2_1_llm_decode/scripts/run_djinn_agents.py \
  --agent-counts 1 2 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 1 \
  --djinn-server localhost:5556 \
  --output-dir Evaluation/exp2_1_llm_decode/results/djinn_agents
```

### Full Experiment (Paper Results)

For the paper, use the config:
```bash
# Ray Keep-Alive: Default --gpu-per-actor 0.01 forces physical OOM (not scheduler queuing)
python Evaluation/exp2_1_llm_decode/scripts/run_ray_keepalive_agents.py \
  --agent-counts 1 2 3 4 \
  --stop-on-oom \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 10

python Evaluation/exp2_1_llm_decode/scripts/run_ray_serverless_agents.py \
  --agent-counts 1 2 4 8 16 32 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 10

python Evaluation/exp2_1_llm_decode/scripts/run_djinn_agents.py \
  --agent-counts 1 2 4 8 16 32 \
  --iterations 1 \
  --new-tokens 50 \
  --sleep-seconds 10 \
  --djinn-server localhost:5556
```

**Note:** The default `--gpu-per-actor 0.01` for Ray Keep-Alive bypasses Ray's logical scheduler and forces physical GPU memory contention, ensuring OOM crashes (not infinite queuing) at low N. This properly demonstrates the "Parking Lot" problem.

### Analysis

Once all three baselines complete:
```bash
python Evaluation/exp2_1_llm_decode/scripts/analyze_agent_scaling.py
```

This generates:
- **Figure 6**: P99 Latency vs Number of Concurrent Agents
- **Table 2**: Summary statistics (mean, p50, p99, OOM thresholds)
- **Claims validation**: Tenant density and latency improvement ratios

### Validation Checklist

- [ ] Ray Keep-Alive OOMs at expected N (2-3 agents) with physical memory exhaustion (not scheduler queuing)
- [ ] Ray Serverless scales to N=32, latency grows >5000ms
- [ ] Djinn scales to N=32, latency stays <200ms
- [ ] Djinn REFLECT latency is significantly lower than REASON latency (proves incremental decode + KV reuse)
- [ ] Djinn session persistence verified (check server logs: "Reusing session")
- [ ] Analysis script correctly computes 20x density and 10x latency claims

### Implementation Notes (OSDI Quality Fixes)

The evaluation includes two critical fixes for scientific rigor:

1. **Incremental Decode for Djinn**: The REFLECT phase sends only new tokens (not full history) to enable true KV cache reuse. This is tracked via `kv_processed_len` in `run_djinn_agents.py`.

2. **Physical OOM for Ray Keep-Alive**: The default `--gpu-per-actor 0.01` bypasses Ray's logical scheduler, forcing physical GPU memory contention. This ensures OOM crashes (not infinite queuing) at low N, properly demonstrating the "Parking Lot" problem.

## Open Tasks
- [ ] Run full agent scaling experiment on production cluster
- [ ] Verify claims (20x density, 10x latency) match expected paper values
- [ ] Generate final figures for OSDI submission

