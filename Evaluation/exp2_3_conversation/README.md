# Experiment 2.3 – Conversational AI (Multi-Turn Dialogue)

Quantifies Djinn’s advantage on session-long conversational workloads.
Conversation consists of a prompt plus nine follow-up turns; we measure latency,
data transfer, and GPU utilization for semantic-blind vs. fully-aware execution,
mirroring `docs/EvaluationPlan.md §6.2`.

## Workload
- **Model:** default `EleutherAI/gpt-j-6b` (override via CLI)
- **Dialogue:** JSON scripted conversation (`prompts/conversation.json`)
- **Turns:** 10 total (1 prompt + 9 responses)
- **Transport:** Client CPU → Djinn server (A100-80GB, 25 Gbps)
- **Semantic feature:** KV cache persistence across turns, phased execution.

## Baselines

| Name | Purpose | Notes |
|------|---------|-------|
| `native_pytorch` | Local generation upper bound | `scripts/run_local_conversation_baseline.py` |
| `semantic_blind` | Djinn with semantics off | future remote harness |
| `full_djinn` | Djinn with semantics on | future remote harness |

Partial baseline omitted to keep runtime manageable; Experiment 2.1 already
covers incremental semantics. This experiment focuses on the dramatic gap
between semantic-blind and full Djinn for session state.

## Metrics
- Latency per turn (ms) + total latency (ms)
- Tokens generated per turn and per-token latency
- Host↔device transfer volume per turn
- GPU utilization (optional NVML)
- KV cache reuse efficiency (Djinn telemetry once wired)

## Files
- `configs/conversation.yaml` – global defaults, run counts
- `prompts/conversation.json` – canonical multi-turn dialogue
- `scripts/run_local_conversation_baseline.py` – HuggingFace runner
- `results/` – JSON dumps for each run/baseline

## How to Run (Local Baseline)
```bash
source .venv/bin/activate
python Evaluation/exp2_3_conversation/scripts/run_local_conversation_baseline.py \
  --model-id distilgpt2 \
  --conversation-file Evaluation/exp2_3_conversation/prompts/conversation.json \
  --device cpu --dtype float32 \
  --runs 1 --warmup-runs 1 \
  --output-dir Evaluation/exp2_3_conversation/results/native_pytorch_smoke
```

Use `distilgpt2` for smoke tests; switch to `EleutherAI/gpt-j-6b` + CUDA for
paper-quality numbers.

## Remote Djinn Conversation
Run the same script with `--backend djinn` to push the multi-turn dialogue through
a remote `privateuseone` device. Provide `--djinn-server host:port` (or use
`GENIE_SERVER_ADDRESS`) and set `DJINN_ENABLE_PRIVATEUSE1_DEVICE=1` before
invoking the script.

```
source .venv/bin/activate
export DJINN_ENABLE_PRIVATEUSE1_DEVICE=1
python Evaluation/exp2_3_conversation/scripts/run_local_conversation_baseline.py \
  --backend djinn \
  --djinn-server localhost:5556 \
  --model-id distilgpt2 \
  --conversation-file Evaluation/exp2_3_conversation/prompts/conversation.json \
  --runs 1 \
  --warmup-runs 0 \
  --output-dir Evaluation/exp2_3_conversation/results/djinn_remote \
  --tag djinn_remote_smoke
```

Compare the `backend` and `djinn_server` fields in the generated JSON to verify
the remote run used Djinn.

## TODO
- [ ] Remote Djinn harness with semantic toggles and telemetry ingestion
- [ ] Analyzer for Figure 8 (latency per turn visualization)

