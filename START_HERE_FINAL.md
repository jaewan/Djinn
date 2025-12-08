# Djinn OSDI Experiment 3 - FINAL EXECUTION GUIDE

**Status**: ✅ Implementation Complete | Ready for H100 Execution  
**Last Updated**: December 8, 2025  
**Branch**: osdi_exp3

---

## Quick Summary

All implementation work for OSDI Experiment 3 is **complete**. The experiment has been upgraded from "toy evaluation" with GPT-2 to a **rigorous scientific study** with Llama-2-13B that will genuinely stress-test Djinn's memory virtualization.

The critical math error (N=6 instead of N=50) has been **identified and fixed**.

---

## What's Been Done

### ✅ Model & Baseline Upgrades
- **Model**: GPT-2 (250MB) → Llama-2-13B (27GB)
- **PyTorch Baseline**: Fixed tokenizer padding, verified VRAM holding (24.3GB)
- **Metrics**: Honest reporting with physical validation

### ✅ Memory Pressure Test Fix
- **Identified Error**: N=6 sessions = 34.8GB (only 43% of H100, no swapping)
- **Corrected**: N=50 sessions = 92GB (exceeds 80GB H100 by 12GB, forces swapping)
- **Validation**: Math verified, explicit logging added

### ✅ Test Script Implementation
- Created: `run_exp3_final_memory_pressure.py`
- Features:
  - Timestamped VRAM tracking per session
  - Swap latency measurement
  - VRAM plateau analysis
  - JSON results export
  - Automatic pass/fail verdict

---

## Files You Need

### Core Experiment Files
1. **Config**: `OSDI_Evaluation/exp3_whitebox_debugging/configs/exp3_osdi_llama.yaml`
   - Model: Llama-2-13B
   - Sessions: 50 (critical!)
   - Breakpoints: [10, 20, 30]

2. **Test Script**: `OSDI_Evaluation/exp3_whitebox_debugging/scripts/run_exp3_final_memory_pressure.py`
   - Main executable for N=50 stress test
   - ~400 lines, well-commented

### Documentation Files
3. **Checklist**: `FINAL_CHECKLIST.md` - What to do next
4. **Feedback Response**: `REVIEWER_2_RESPONSE.md` - Math explanation
5. **Improvements Summary**: `OSDI_EXP3_IMPROVEMENTS.md` - Quality upgrades

---

## The Critical Math

```
N=50 Sessions on H100:
  Weights: 27GB (Llama-2-13B, shared once)
  KV cache: 50 × 1.3GB = 65GB
  Total: 92GB > 80GB capacity

Expected Result:
  ✅ Sessions 1-40: Fit in GPU
  ✅ Sessions 41-50: Trigger swap to host RAM
  ✅ VRAM stays < 80GB (proves swapping)
  ✅ All 50 sessions complete
```

---

## How to Run

### Prerequisites
- H100 GPU with 80GB VRAM available
- Python 3.8+ with Djinn installed
- Virtual environment activated

### Step 1: Start Djinn Server
```bash
cd /home/ubuntu/Djinn
source .venv/bin/activate
python3 -m djinn.server.server_main --port 5556 --gpu 0
```

Wait for log message: `Server listening on 0.0.0.0:5556`

### Step 2: Run Test (in new terminal)
```bash
cd /home/ubuntu/Djinn
source .venv/bin/activate
python3 OSDI_Evaluation/exp3_whitebox_debugging/scripts/run_exp3_final_memory_pressure.py \
  --output-dir /tmp/exp3_final_results \
  --num-sessions 50
```

### Step 3: Wait for Results
- Runtime: ~5-10 minutes
- Watch for output like:
  ```
  [Session  1/50] 01:23:45
    VRAM before: 27.45GB
    VRAM after:  28.73GB
    Spawn time: 45.2ms
  
  [Session 41/50] 01:24:30
    ⚠️  SWAP DETECTED: ~65.3ms (session 1 triggered eviction)
  ```

---

## What You're Looking For

### Success Indicators
1. **All 50 sessions complete**: No OOM errors
2. **VRAM plateau**: Peak < 80GB (typically 75-78GB)
3. **Swap latencies**: Sessions 41+ show 50-80ms latency
4. **Clean completion**: Script finishes with JSON results

### Results File
Location: `/tmp/exp3_final_results/memory_pressure_results.json`

Should look like:
```json
{
  "status": "success",
  "num_sessions_requested": 50,
  "num_sessions_spawned": 50,
  "vram_stats": {
    "min_gb": 27.45,
    "max_gb": 77.82,
    "avg_gb": 65.3
  },
  "swapping_active": true,
  "vram_progression": [
    {"session": 1, "vram_gb": 28.73, "timestamp": "01:23:45"},
    ...
    {"session": 50, "vram_gb": 76.21, "timestamp": "01:24:35"}
  ]
}
```

---

## Git Commits to Know About

| Commit | What |
|--------|------|
| e2a33a1 | OSDI Exp3: Complete Quality Upgrade |
| 57cb5c2 | CRITICAL FIX: N=6 → N=50 |
| 22d1ace | Document Reviewer #2 Feedback |
| f529286 | Final implementation: Test script ready |

All changes are in the `osdi_exp3` branch and properly tracked.

---

## After Test Completes

1. **Verify results**: Check that VRAM peaked below 80GB
2. **Save logs**:
   ```bash
   cp /tmp/exp3_final_results/exp3_memory_pressure_final.log \
      /home/ubuntu/Djinn/OSDI_Evaluation/exp3_whitebox_debugging/results/
   ```
3. **Commit results**:
   ```bash
   cd /home/ubuntu/Djinn
   git add OSDI_Evaluation/exp3_whitebox_debugging/results/
   git commit -m "✅ Exp3 Final Results: N=50 memory pressure test complete"
   ```
4. **Prepare for submission**: Logs are now ready for OSDI reviewers

---

## Reviewer #2's Final Verdict

> "Once you have that log file, you are done."

The log file will contain exactly what Reviewer #2 needs:
- ✅ Timestamped VRAM progression (shows plateau below 80GB)
- ✅ All 50 sessions completed (no OOM)
- ✅ Swap latencies measured (~50-80ms, PCIe-bound)
- ✅ Math validated (92GB exceeds 80GB capacity)

---

## Troubleshooting

### Server won't start
```bash
pkill -f server_main
sleep 2
python3 -m djinn.server.server_main --port 5556 --gpu 0
```

### Connection refused (localhost:5556)
- Wait longer for server to start
- Check: `ps aux | grep server_main`
- Check logs: `tail -50 /tmp/djinn_server_final.log`

### OOM during test
- GPU already has other processes: Kill and restart
- Djinn server issues: Restart server cleanly
- Model loading: Verify Llama-2-13B is accessible

### VRAM not plateauing
- Check if sessions are actually being paused
- Verify `pause_layer: 20` in config
- Check Djinn server logs for errors

---

## Contact Points

**Questions about**:
- **Math/Design**: See `REVIEWER_2_RESPONSE.md`
- **What was upgraded**: See `OSDI_EXP3_IMPROVEMENTS.md`
- **Execution steps**: See `FINAL_CHECKLIST.md`
- **Test details**: See script comments in `run_exp3_final_memory_pressure.py`

---

## Next Steps Summary

1. ✅ Code implementation: COMPLETE
2. ⏳ **Start Djinn server**
3. ⏳ **Run the test script**
4. ⏳ Verify results show VRAM < 80GB
5. ⏳ Commit results to git
6. ⏳ Prepare paper for OSDI submission

**Estimated total time**: ~20 minutes from now to complete submission

---

## Success Criteria (OSDI Ready)

| Criterion | Status | Evidence |
|-----------|--------|----------|
| Model: Llama-2-13B | ✅ | Config specifies `meta-llama/Llama-2-13b-hf` |
| Math: N=50 = 92GB | ✅ | Validated in code, logged during test |
| Baseline: PyTorch | ✅ | Shows 24.3GB VRAM held during pause |
| Test: All 50 complete | ⏳ | After running test |
| VRAM < 80GB | ⏳ | After running test |
| Swap latency: 50-80ms | ⏳ | After running test |
| Logs available | ⏳ | Will be at `/tmp/exp3_final_results/` |

---

## Summary

**What changed from initial submission**: 
- Model upgraded (GPT-2 → Llama-2-13B) to show real resource pressure
- Math corrected (N=6 → N=50) to actually exceed GPU capacity
- Test script enhanced to prove swapping is working
- Documentation clarified for Reviewer #2

**Current state**: 
- ✅ All code ready
- ✅ Configuration correct  
- ✅ Test script implemented
- ⏳ Awaiting execution on H100

**Confidence level**: 95% that test will show swapping works and pass all criteria

---

**Ready to execute?** Run the command in "Step 2" above and watch the VRAM plateau!
