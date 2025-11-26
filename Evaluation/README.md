# Djinn Evaluation Workspace

This directory hosts runnable artifacts for the OSDI evaluation plan described in
`docs/EvaluationPlan.md`.  Each experiment lives in its own subdirectory with
configs, scripts, and raw result storage so we can iterate without touching the
core Djinn tree.

## Structure
- `exp2_1_llm_decode/` – Experiment 2.1 (hero evaluation). Future experiments
  will follow the same layout (`configs/`, `scripts/`, `results/`, `notes/`).

## Workflow
1. **Plan** – translate the requirements from `docs/EvaluationPlan.md` into a
   per-experiment README that specifies workload, metrics, and pass criteria.
2. **Implement** – add scripts in `scripts/` plus any helper configs/templates.
3. **Run** – exercise the harnesses (e.g.,
   `python Evaluation/exp3_2_memory_kernel/scripts/run_memory_kernel.py --config Evaluation/exp3_2_memory_kernel/configs/session_stress.yaml --allocator vmu --duration 30 --output Evaluation/exp3_2_memory_kernel/results/vmu_full.json`)
   for real GPU-backed metrics, and write JSON/CSV into
   `results/<baseline>/<timestamp>.json`.
4. **Analyze** – notebooks or scripts emit the figures referenced in the plan.

## Running an evaluation
- Activate the workspace environment (`source .venv/bin/activate` or equivalent).
- Pick the experiment subdirectory (e.g., `exp3_2_memory_kernel/`), read its README,
  and invoke the script under `scripts/`. Each script documents the allocator,
  duration, and output format; run it with `--help` to see the arguments.
- Results land in `Evaluation/<exp>/results/…`. Keep the JSON/CSV files for further
  plotting, and add summaries or derived figures to the subdirectory’s `notes/`
  or `figures/` folder once they stabilize.

## Environment
- Python ≥3.10
- PyTorch + CUDA toolchain that matches Djinn runtime
- Hugging Face `transformers` for baseline workloads
- Optional: `pynvml` for GPU utilization sampling (see experiment README)

Each script declares its own requirements; prefer `pip install -r
requirements-dev.txt` at the repo root before running evaluations.

## Contributing
Keep experiment-specific assets confined to their directory, add short docstrings
to each script, and store raw outputs (not checked-in) under `results/`.  Derived
tables/figures should be checked into `notes/` or `figures/` once finalized to
keep provenance clear.

---
## Evaluation Readiness Guide (A100‑40GB environment)

This document summarizes the evaluation suite, the exact pre-work required before running on the production cluster, and launch commands/scripts for each experiment. It follows the structure in `docs/EvaluationPlan.md`.

---

### Shared preparation (run once per node)

1. **Environment setup**
   - `cd /home/jae/Genie`
   - `python3 -m venv .venv && source .venv/bin/activate`
   - `pip install -r requirements.txt`
   - Install extras used in diagnostics: `pip install xxhash prometheus-client`
   - Export convenience env vars (customize as needed):
     ```bash
     export GENIE_SERVER_ADDRESS=localhost:5556
     export GENIE_QOS_MAX_CONCURRENCY=12
     export GENIE_QOS_CLASS_SHARES="realtime=0.1,interactive=0.7,batch=0.2"
     export GENIE_QOS_ESCALATION_DELAY_MS=300
     export GENIE_EVAL_MIN_FREE_GB=10
     ```

2. **Model cache priming**
   - Download large HuggingFace models ahead of time to avoid counting download time:
     ```
     python - <<'PY'
     from transformers import AutoModelForCausalLM, AutoTokenizer
     models = [
         "meta-llama/Llama-2-7b-hf",
         "meta-llama/Llama-2-13b-hf",
         "EleutherAI/gpt-j-6B",
         "openai/whisper-large",
         "facebook/convnext-base-384",
         "openai/clip-vit-base-patch32"
     ]
     for mid in models:
         AutoModelForCausalLM.from_pretrained(mid, device_map="cpu")
         AutoTokenizer.from_pretrained(mid)
     PY
     ```
   - For vision/audio models, run the analogous download with `AutoModelForImageClassification`, `AutoModel`, `AutoProcessor`, etc.
   - Store checkpoints on the server’s NVMe; ensure they’re readable by Djinn.

3. **Server boot**
   - On each evaluation node, start Djinn once:
     ```bash
     source .venv/bin/activate
     GENIE_QOS_MAX_CONCURRENCY=12 \
     GENIE_QOS_CLASS_SHARES="realtime=0.1,interactive=0.7,batch=0.2" \
     GENIE_QOS_ESCALATION_DELAY_MS=300 \
     python -m djinn.server.server_main \
         --node-id prod-a100 \
         --control-port 5555 \
         --data-port 5556 \
         --gpus 0  # add more indices if using multi-GPU
     ```
   - Keep server logs (`/tmp/djinn_server.log`) for traceability.

4. **Metrics/plotting directories**
   - Ensure `Evaluation/*/results` exist and writable.
   - Set `GENIE_EVAL_METRICS_PATH` if collecting VMU diagnostics: `export GENIE_EVAL_METRICS_PATH=/data/logs/vmu_metrics.json`.

---

### Experiment 2.1 – LLM Decode (hero experiment)

**Purpose:** Show semantic awareness transforms disaggregation (Figures 6a–c).

1. **Pre-download**: `meta-llama/Llama-2-7b-hf`, tokenizer, plus semantic-blind config.
2. **Baselines**: `native_pytorch`, `semantic_blind`, `partial`, `full`.
3. **Config**: `Evaluation/exp2_1_llm_decode/configs/baselines.yaml`.
4. **Run**:
   ```bash
   source .venv/bin/activate
   python Evaluation/exp2_1_llm_decode/scripts/run_local_baseline.py \
       --config Evaluation/exp2_1_llm_decode/configs/baselines.yaml \
       --runs 30 --server localhost:5556
   ```
5. **Outputs**: JSON/CSV per baseline in `Evaluation/exp2_1_llm_decode/results`. Use plotting notebook to generate Fig 6a–c (ensure data transfer stats recorded via transport metrics).

---

### Experiment 2.2 – Streaming Audio (Whisper)

1. **Pre-download**: `openai/whisper-large-v2`, audio assets.
2. **Baseline config**: `Evaluation/exp2_2_streaming_audio/configs/streaming_config.yaml`.
3. **Run**:
   ```bash
   python Evaluation/exp2_2_streaming_audio/scripts/run_local_streaming_baseline.py \
       --config Evaluation/exp2_2_streaming_audio/configs/streaming_config.yaml \
       --server localhost:5556 --runs 30
   ```
4. **Notes**: Ensure audio chunks cached locally; warm up server to avoid first-run JIT cost.

---

### Experiment 2.3 – Conversational AI (GPT-J)

1. **Pre-download**: `EleutherAI/gpt-j-6B`.
2. **Config**: `Evaluation/exp2_3_conversation/configs/conversation.yaml`.
3. **Run**:
   ```bash
   python Evaluation/exp2_3_conversation/scripts/run_local_conversation_baseline.py \
       --config Evaluation/exp2_3_conversation/configs/conversation.yaml \
       --server localhost:5556 --runs 30
   ```

---

### Experiment 3.1 – Component Ablation

1. **Config**: `Evaluation/exp3_1_ablation/configs/ablation_matrix.yaml`.
2. **Smoke test**: 
   ```bash
   python Evaluation/exp3_1_ablation/scripts/render_matrix.py \
       --matrix Evaluation/exp3_1_ablation/configs/ablation_matrix.yaml \
       --workloads Evaluation/exp3_1_ablation/configs/workloads.yaml
   ```
3. **Run**: once run harness is wired (reuse Exp 2.1 runner, toggling feature flags). Expect 6 configs × 2 workloads × 30 runs. Save to `Evaluation/exp3_1_ablation/results`.

---

### Experiment 3.2 – Memory Kernel Deep Dive

1. **Config**: `Evaluation/exp3_2_memory_kernel/configs/session_stress.yaml`.
2. **Synthetic quick run**:
   ```bash
   python Evaluation/exp3_2_memory_kernel/scripts/run_memory_stress.py \
       --config Evaluation/exp3_2_memory_kernel/configs/session_stress.yaml \
       --allocator vmu --duration 30 \
       --output Evaluation/exp3_2_memory_kernel/results/vmu_synth.json
   ```
3. **24‑hour real run**:
   ```bash
   python Evaluation/exp3_2_memory_kernel/scripts/run_memory_kernel.py \
       --config Evaluation/exp3_2_memory_kernel/configs/session_stress.yaml \
       --allocator vmu --duration 86400 \
       --output Evaluation/exp3_2_memory_kernel/results/vmu_real.json
   ```
   Repeat with `--allocator torch`.
4. **Plot**: `scripts/analyze_memory_kernel.py` (to be added) for Fig 9 / Table 3.

---

### Experiment 4.1 – Scalability

1. **Config**: `Evaluation/exp4_1_scalability/configs/load_sweep.yaml`.
2. **Run**:
   ```bash
   python Evaluation/exp4_1_scalability/scripts/run_load_sweep.py \
       --config Evaluation/exp4_1_scalability/configs/load_sweep.yaml \
       --server localhost:5556
   ```
   Configure durations/concurrency to match [1, 2, 4, 8, 16, 32, 64] users. Ensure GPU metrics sampler is active.

---

### Experiment 4.2 – QoS guarantees

1. **Config**: `Evaluation/phase4_load_test/configs/production_load_test.yaml` (or smoke variant).
2. **Preload**: ResNet-50, ViT-base, CLIP, GPT‑2, etc.
3. **Run**:
   ```bash
   python Evaluation/phase4_load_test/scripts/run_production_load_test.py \
       --config Evaluation/phase4_load_test/configs/production_load_test.yaml
   ```
   Capture per-class P99, SLA violations. For FCFS baseline, disable QoS via env (`GENIE_ENABLE_QOS=0`).

---

### Experiment 5.1 – Overhead analysis

1. **Config**: start with `Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml`, extend to full suite (BERT, GPTs, etc.).
2. **Pre-download**: All model sizes listed in §6.5.
3. **Run**:
   ```bash
   python Evaluation/exp5_1_overhead/scripts/run_overhead_sweep.py \
       --config Evaluation/exp5_1_overhead/configs/overhead_hf_smoke.yaml \
       --tag hf_smoke_run --workloads hf_tiny_gpt2
   ```
   For full sweep, point config to expanded list, increase `experiment.runs` to 30, and ensure both semantic/blind baselines use true remote executions (no scaling placeholders). Use `analyze_overhead.py` to produce Figure 12.

---

### Experiment 6.1 – Cross-workload generality

1. **Config**: `Evaluation/exp6_1_generality/configs/generality_smoke.yaml`.
2. **Run**:
   ```bash
   python Evaluation/exp6_1_generality/scripts/run_generality_suite.py \
       --config Evaluation/exp6_1_generality/configs/generality_smoke.yaml \
       --server localhost:5556
   ```
   Expand workload list to include six model families; capture semantic efficiency ratio.

---

### Data handling & plotting

- Each experiment writes JSON manifests under its `results/` directory.
- Plotting notebooks / scripts (e.g., `analyze_overhead.py`, future `analyze_ablation.py`) should be executed on the same node to avoid path issues.
- Back up raw data before regenerating figures; include `git lfs` if storing large CSVs.

---

### Final checklist before running on production GPU

1. ✅ All HuggingFace models downloaded and cached.  
2. ✅ Djinn server running with desired QoS settings.  
3. ✅ `GENIE_SERVER_ADDRESS` exported on client shells.  
4. ✅ Metrics/log directories writable.  
5. ✅ For long experiments (e.g., memory kernel, scalability), ensure persistent logging (e.g., `tee` server logs).  
6. ✅ Validate smoke tests on each script before committing to 30-run full experiments.  
7. ✅ Document exact commands and env vars in a runbook for reproducibility/artifact package.

This guide captures the prep and execution steps for each evaluation in `docs/EvaluationPlan.md`. Update as configs evolve (e.g., once ablation runner is fully automated).