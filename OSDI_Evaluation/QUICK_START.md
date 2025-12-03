# Quick Start: Skip-End Ring Buffer Evaluation

**Time to First Result**: ~30 minutes (smoke test) or 2-3 hours (full experiment)

---

## 1. Setup Environment

```bash
cd /home/jhong/Djinn
source .venv/bin/activate

# Install dependencies (if not already done)
pip install -r requirements.txt
pip install pyyaml transformers pynvml
```

---

## 2. Validate Pinned Memory (5 min)

```bash
# This must show >22 GB/s for Host→Device (Pinned)
python benchmarks/shm_bandwidth.py

# Expected output:
# ✅ Host → Device (Pinned):    24.3 GB/s (pinned)
# ✅ PASS: Pinned bandwidth 24.3 GB/s >= threshold 22.0 GB/s
```

If you get **low bandwidth (<10 GB/s)**:
```bash
# Enable pinned memory
ulimit -l unlimited

# Disable OS swap
sudo swapoff -a

# Try again
python benchmarks/shm_bandwidth.py
```

---

## 3. Smoke Test (GPT-J, 15 min)

### 3a. Download Model

```bash
python OSDI_Evaluation/exp2_virtual_memory/scripts/download_model.py \
    --model EleutherAI/gpt-j-6B
```

### 3b. Start Djinn Server

In **Terminal 1**:
```bash
export GENIE_VMU_RING_BUFFER=1
export GENIE_VMU_RING_BUFFER_GB=12

python -m djinn.server.server_main \
    --node-id test-node \
    --control-port 5555 \
    --data-port 5556 \
    --gpus 0
```

Wait for: `✅ Server ready, listening on 0.0.0.0:5556`

### 3c. Run Smoke Test

In **Terminal 2**:
```bash
cd /home/jhong/Djinn

python OSDI_Evaluation/exp2_virtual_memory/scripts/run_virtual_memory_experiment.py \
    --config OSDI_Evaluation/exp2_virtual_memory/configs/virt_mem_smoke.yaml \
    --runs 2 \
    --output OSDI_Evaluation/exp2_virtual_memory/results/smoke_test.json
```

**Expected output**:
```
Run 0: latency 1250.4ms, bandwidth 4.8GB/s, memory 5234MB
Run 1: latency 1238.1ms, memory 5201MB
✅ PASS: Bandwidth sustained > 20 GB/s
```

---

## 4. Full Experiment (Llama-70B, 2-3 hours)

### 4a. Download Llama-70B

```bash
# This may take 30-60 min depending on internet speed
python OSDI_Evaluation/exp2_virtual_memory/scripts/download_model.py \
    --model meta-llama/Llama-2-70b-hf
```

### 4b. Update Server Configuration

In **Terminal 1**, restart server with:
```bash
export GENIE_VMU_RING_BUFFER=1
export GENIE_VMU_RING_BUFFER_GB=48  # 48GB ring buffer
export GENIE_VMU_RING_BUFFER_WORKERS=1

python -m djinn.server.server_main \
    --node-id osdi-eval \
    --control-port 5555 \
    --data-port 5556 \
    --gpus 0
```

### 4c. Run Full Experiment

In **Terminal 2**:
```bash
python OSDI_Evaluation/exp2_virtual_memory/scripts/run_virtual_memory_experiment.py \
    --config OSDI_Evaluation/exp2_virtual_memory/configs/virt_mem_config.yaml \
    --model meta-llama/Llama-2-70b-hf \
    --runs 5 \
    --output OSDI_Evaluation/exp2_virtual_memory/results/llama_70b_full.json
```

This will:
1. Load 140GB Llama-70B model (takes ~1-2 min)
2. Register model with ring buffer
3. Install weight hooks
4. Run 2 warmup iterations (measured)
5. Run 5 measurement iterations (takes ~40 min total)
6. Print bandwidth and latency results

**Expected output**:
```
Run 0: latency 6850ms, bandwidth 20.4GB/s, memory 58200MB
Run 1: latency 6920ms, bandwidth 20.2GB/s, memory 58150MB
...
✅ PASS: Bandwidth sustained > 20 GB/s
```

---

## 5. Analyze Results

```bash
python OSDI_Evaluation/exp2_virtual_memory/scripts/analyze_results.py \
    --results OSDI_Evaluation/exp2_virtual_memory/results/ \
    --output OSDI_Evaluation/exp2_virtual_memory/figures/
```

Output:
```
VIRTUAL MEMORY RING BUFFER EXPERIMENT RESULTS
==============================================

Result: llama_70b_full
  Model: meta-llama/Llama-2-70b-hf (140.0GB)
  Runs: 5

  Effective Bandwidth:
    Average:   20.4 GB/s
    Median:    20.3 GB/s
    Min/Max:   19.8 / 21.1 GB/s

  Status: ✅ PASS
  → Bandwidth 20.4 GB/s >= threshold 20.0 GB/s
```

---

## Key Metrics

### What to Check

| Metric | Target | Pass Criteria |
|--------|--------|---------------|
| Pinned Memory BW | >24 GB/s | Baseline validation |
| Ring Buffer BW | >20 GB/s | **Primary success metric** |
| Latency | <10s for 512→50 tokens | Performance sanity |
| VRAM Usage | <60GB | Fits in GPU memory |
| Logit Match | Norm < 0.1 | Correctness validation |

### Performance Interpretation

- **>20 GB/s**: ✅ Excellent - PCIe is saturated, pipelining works
- **15-20 GB/s**: ⚠️ Acceptable - Some prefetch overhead, but acceptable
- **<15 GB/s**: ❌ Poor - Prefetch thread not keeping up

---

## Troubleshooting

### Problem: "Low bandwidth (< 15 GB/s)"

**Solutions** (try in order):
1. Check if prefetch thread is running:
   ```bash
   # Add debug logging to streamer
   # Look for "✅ Prefetch" messages in logs
   ```

2. Verify pinned memory:
   ```bash
   ulimit -l unlimited
   swapoff -a
   ```

3. Check NUMA binding:
   ```bash
   # If multi-socket, bind to GPU-local CPU:
   numactl --cpunodebind=0 --membind=0 python ...
   ```

4. Disable Python GC during measurement:
   ```bash
   # Already set in config: disable_gc_during_run: true
   ```

### Problem: "CUDA Illegal Memory Access"

**Cause**: Ring buffer wrap logic failed (tensor split across buffer boundary)

**Debug**:
1. Add print statements in `ring_buffer.py` at register time:
   ```python
   logger.info(f"Layer {i}: offset={alloc.offset}, size={alloc.size_bytes}")
   ```

2. Verify skip-end logic:
   ```python
   # After register, all offsets + sizes should fit before wrap
   assert all(alloc.offset + alloc.size_bytes <= CAPACITY)
   ```

### Problem: "Model not found" or "CUDA OOM"

**Solutions**:
1. Pre-download models (see step 4a)
2. Reduce batch size in config
3. Use smaller model for testing (GPT-J instead of Llama-70B)

---

## Files to Monitor

During experiment, watch these log files:

```bash
# Server logs (Terminal 1)
tail -f /tmp/djinn_server.log

# Client experiment logs (Terminal 2)
tail -f /tmp/djinn_virt_mem_exp.log

# Memory usage
watch -n 1 nvidia-smi
```

---

## Configuration Tuning

If needed, edit `configs/virt_mem_config.yaml`:

```yaml
# To use smaller ring buffer:
ring_buffer:
  capacity_gb: 32  # Instead of 48

# To run fewer iterations:
measurement:
  runs: 2  # Instead of 5

# To disable async prefetch (for ablation):
baselines:
  ring_buffer_async_prefetch:
    use_async_prefetch: false
```

---

## Timeline

| Step | Time | Notes |
|------|------|-------|
| Setup | 5 min | Environment activation |
| Validate pinned mem | 5 min | Quick BW check |
| Smoke test (GPT-J) | 15 min | Validates end-to-end |
| Download Llama-70B | 30-60 min | Network dependent |
| Full experiment | 45 min | 2 warmup + 5 measure |
| Analysis | 5 min | Generate reports |
| **Total** | **2-3 hours** | From zero to results |

---

## Success Criteria

✅ **Experiment Passes** if:
- [x] Ring buffer registers model successfully
- [x] Hooks install without errors
- [x] Inference completes without CUDA errors
- [x] Effective bandwidth > 20 GB/s (average over 5 runs)
- [x] VRAM usage < 60GB (fits in GPU)
- [x] Output logits match reference within FP16 tolerance

---

## Next: Paper Figure Generation

After successful experiment, results support Figures 7a-7c in the paper:
- **Figure 7a**: Effective bandwidth comparison (ring buffer vs baselines)
- **Figure 7b**: Timeline showing prefetch + compute overlap
- **Figure 7c**: VRAM utilization curve during inference

See `IMPLEMENTATION_SUMMARY.md` for full analysis workflow.

