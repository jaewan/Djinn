# Experiment 2.2 – Streaming Audio Transcription

Demonstrates Djinn’s advantage on sequential, stateful workloads. We chunk a
30-second audio clip into overlapping 5-second windows, transcribe in real time,
and compare four baselines (native, semantic-blind, partially-aware, full
Djinn). Goal: show 34× latency reduction and 3,750× data savings versus
semantic-blind execution as defined in `docs/EvaluationPlan.md §6.2`.

## Workload
- **Model:** default `openai/whisper-large-v3` (override via CLI)
- **Input:** 30s mono wav at 16 kHz (place under `data/long_sample.wav`)
- **Chunking:** 5s window, 1s stride (configurable)
- **Transport:** client CPU → Djinn server (A100-80GB, 25 Gbps)
- **Semantic optimizations:** state persistence across chunks, skeletonization,
  streaming KV cache.

## Baselines

| Name | Description | Notes |
|------|-------------|-------|
| `native_pytorch` | Local Whisper transcription (upper bound) | `scripts/run_local_streaming_baseline.py` |
| `semantic_blind` | Djinn with semantics disabled | Refer to `configs/baselines.yaml` |
| `partially_aware` | Djinn caching + skeletonization, no chunk state | same |
| `full_djinn` | All semantic features enabled | same |

Baseline toggles align with Experiment 2.1; they are shared in
`configs/baselines.yaml`.

## Metrics
- Chunk latency (ms) and cumulative latency for 30 s stream
- Throughput (audio seconds processed per wall-second)
- Data transferred host↔device per chunk (MB)
- GPU SM utilization (%), memory bandwidth (optional NVML)
- Queue / compute / transfer breakdown (Djinn telemetry)

## Artifacts
- `configs/streaming_config.yaml` – chunk/window parameters, IO paths
- `scripts/run_local_streaming_baseline.py` – native PyTorch reference
- `results/` – JSON payloads, one per baseline run
- `notes/` – derived figures (Figure 7a–7c) once analysis script lands

## How to Run (Local Baseline)
```bash
source .venv/bin/activate
pip install torchaudio soundfile

python Evaluation/exp2_2_streaming_audio/scripts/run_local_streaming_baseline.py \
  --model-id openai/whisper-base \
  --audio-file Evaluation/exp2_2_streaming_audio/data/long_sample.wav \
  --chunk-seconds 5 \
  --stride-seconds 1 \
  --runs 3 \
  --output-dir Evaluation/exp2_2_streaming_audio/results/native_pytorch \
  --sample-gpu
```

For smoke testing without an audio file, pass `--dummy-audio 10` to synthesize a
10-second sine wave clip.

## Remote Djinn Streaming
Once the Djinn server is up, switch to the remote backend with `--backend djinn`
and point at the service via `--djinn-server` or `GENIE_SERVER_ADDRESS`.
Make sure `DJINN_ENABLE_PRIVATEUSE1_DEVICE=1` is exported so the `privateuseone`
backend is registered before the workload runs.

```
source .venv/bin/activate
export DJINN_ENABLE_PRIVATEUSE1_DEVICE=1
python Evaluation/exp2_2_streaming_audio/scripts/run_local_streaming_baseline.py \
  --backend djinn \
  --djinn-server localhost:5556 \
  --dummy-audio 5 \
  --runs 1 \
  --warmup-runs 0 \
  --tag djinn_remote_smoke \
  --output-dir Evaluation/exp2_2_streaming_audio/results/djinn_remote
```

Watch the per-chunk JSON for `backend` and `djinn_server` metadata and confirm the
Djinn logs show remote transfers (or fallback warnings if the server is unreachable).

## TODO
- [ ] Djinn remote harness with semantic toggles (mirrors Exp 2.1)
- [ ] Analyzer that fuses per-chunk telemetry into Figures 7a–7c
- [ ] Add curated 30 s reference audio clip (licensing-friendly)

