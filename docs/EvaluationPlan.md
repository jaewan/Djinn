# OSDI Evaluation Plan & Execution Checklist: Djinn (v2.3.15)

**Status:** **APPROVED FOR EXECUTION**
**Target Venue:** OSDI / SOSP
**Core Thesis:** Djinn is a **Tensor Operating System**. By lifting OS primitives (Virtualization, Paging, Scheduling) into the framework layer, it enables **Interactivity** and **High-Density Time-Sharing** where rigid Batch (Ray) and Serving (vLLM) systems fail.

---

## 1. Environment & Configuration

*   **Hardware:** Single Node, 1x NVIDIA A100-80GB (PCIe Gen4).
    *   **Artificial Constraint:** We strictly enforce a **60GB VRAM Limit** for the Djinn VMU. This reserves 20GB for system overhead and forces paging behavior early.
*   **Host Memory:** 1TB System RAM.
    *   **CRITICAL:** `GENIE_PINNED_MEMORY=1`. All host-side weight buffers must utilize `cudaHostAlloc` (Pinned Memory) via PyTorch’s `pin_memory()`. Standard pageable memory will bottleneck bandwidth at 4GB/s; we need >24GB/s.
    *   **OS Config:** `swapoff -a` (Disable OS Swap) to prevent kernel-level thrashing.
    *   **Ulimit:** `ulimit -l unlimited` (Allow locking unlimited memory pages).
*   **Baselines:**
    *   **vLLM (Reactive):** Configured with standard block settings. Represents "Hardware-unaware LRU Paging."
    *   **Ray (Static):** 1 Actor per Agent. Represents "Static Partitioning."
    *   **HuggingFace Accelerate (Offloading):** Configured with `device_map="auto"` and `offload_folder`. Represents standard "Zero-3" style offloading.
    *   **Djinn (Semantic):** v2.3.15 with Unified VMU, SRG Scheduler, and **Skip-End Ring Buffer**.

---

## 2. Implementation Specification: The "Skip-End Ring Buffer"

**Architectural Role:** Software-defined MMU for the `Text Segment` (Weights).
**Goal:** Saturate PCIe Gen4 Bandwidth (>24 GB/s) to enable the "Infinite Memory" illusion.

### 2.1 Memory Partitioning
We explicitly partition the 60GB VRAM budget.
*   **Total Budget:** 60GB
*   **Partition A: Weight Ring Buffer (48GB)**
    *   **Structure:** Circular Buffer, pre-allocated as a single ByteTensor.
    *   **Allocation Strategy:** **"Skip-End" Allocator** (Deterministic Pre-Calc).
        *   *Constraint:* CUDA Tensors must be physically contiguous. We **CANNOT** split a single weight matrix across the end/start of the buffer.
        *   *Logic:* Offsets are calculated **once at startup**, not at runtime.
            ```python
            # Startup Phase
            current_offset = 0
            for layer in model:
                if (current_offset + layer.size > RING_SIZE):
                    current_offset = 0  # Skip the tail, wrap to start
                layer.ring_offset = current_offset
                current_offset += layer.size
            ```
*   **Partition B: Unified Heap (12GB)**
    *   Structure: Contiguous Slab.
    *   Usage: `Stack Slab` (Activations) + `Data Segment` (KV Cache).
    *   *Swap Trigger:* If KV usage exceeds 12GB, the Scheduler strictly swaps an idle session to Host.

### 2.2 The Runtime Logic (Async Pipelining)
**Crucial Change:** No CPU blocking (`synchronize()`) inside the loop. Fully asynchronous GPU-side event dependency.

*   **Streams:**
    *   `Stream A` (Prefetcher): High-priority copy stream (Host $\to$ Device).
    *   `Stream B` (Compute): Kernel execution stream.
*   **The Logic (Per Layer):**
    1.  **CPU (Hook):**
        *   *Optimization:* Do not create views dynamically. Use pre-computed views.
        *   `module.weight.data = self.precomputed_views[layer_idx]`
        *   *Note:* This happens on CPU. It updates metadata only (O(1)).
    2.  **Prefetcher (Stream A):**
        *   `stream_a.wait_event(Event_Layer_N_3_Done)` (Ensure we don't overwrite a slot still in use).
        *   `cudaMemcpyAsync(dst=ring_buffer_ptr + offset, src=host_weight_ptr)`.
        *   `Event_Layer_N_Ready.record(stream_a)`.
    3.  **Executor (Stream B):**
        *   `stream_b.wait_event(Event_Layer_N_Ready)` (Wait for data).
        *   `Layer_N.forward()` (Kernel launch).
        *   `Event_Layer_N_Done.record(stream_b)` (Signal slot is free).

---

## 3. Experiments

### Experiment 1: The "Semantic Scheduling" Test (Agent Scalability)
**Scientific Goal:** Prove that **Semantic Awareness** (knowing *when* to swap) outperforms **Reactive Heuristics** (LRU).

*   **Workload:** RAG Agent Loop ($N$ Concurrent Agents).
    *   `Prefill (2k tokens)` $\to$ `Decode (50 tokens)` $\to$ **`Tool Wait (10s)`** $\to$ `Resume`.
*   **The Stressor:** 50 Agents $\times$ 2GB KV Cache = 100GB. (Exceeds 60GB VMU Limit).
*   **The Comparison:**
    *   **vLLM:** Holds KV in VRAM during "Tool Wait" (LRU policy). When 50 agents resume, it thrashes or blocks new requests.
    *   **Djinn:** Detects `Tool Wait` via SRG (gap in graph). **Proactively** swaps KV to Host Pinned Memory.
*   **Metrics:**
    1.  **Resume Latency (P99):** Time from "Tool Return" to "First Token." (Must include PCIe transfer time).
    2.  **Throughput Sustainability:** Tokens/sec vs Concurrent Agents. (vLLM should drop to near-zero; Djinn should maintain throughput with added latency offset).
    3.  **Success Rate:** 100% completion (vs. Client Timeout/OOM in baselines).

### Experiment 2: The "Virtual Memory" Test (Heterogeneous Pipelines)
**Scientific Goal:** Prove the **Ring Buffer** enables running models larger than physical VRAM by saturating PCIe.

*   **Workload:** Llama-3-70B Inference (140GB Weights, FP16) on 60GB VRAM.
*   **Constraint:** `Batch Size = 1`.
*   **The Comparison:**
    *   **vLLM:** **Crash (OOM).** Cannot page weights dynamically.
    *   **HF Accelerate:** **Slow.** Python dispatch overhead + synchronous copies lead to gaps in PCIe usage.
    *   **Djinn:** **Fast.** The Ring Buffer hides the latency of Layer $N$ transfer behind the compute of Layer $N-1$.
    *   **Djinn (No-Prefetch Ablation):** **Medium.** Same Ring Buffer, but synchronous copy. Proves the value of Async Pipelining.
*   **Metrics:**
    1.  **Effective Bandwidth:** $\frac{\text{Model Size}}{\text{Total Inference Time}}$. Target: **>20 GB/s**.
    2.  **Visual Proof:** `nvidia-smi` trace showing flatlined PCIe RX and continuous Compute utilization.

### Experiment 3: The "White-Box" Test (Intervention)
**Scientific Goal:** Show `LazyTensor` abstraction allows zero-cost context switching.

*   **Workload:** Breakpoint Debugging.
    *   Run Layer 1-15 $\to$ Pause (User copies tensor to CPU) $\to$ Run Other Job $\to$ Resume Layer 16.
*   **Metric:** **System Overhead** (Time spent saving/restoring the `Stack Slab` vs. Compute Time).

---

## 4. The Execution Checklist (Junior Engineer Guide)

**Instructions:** Check off items sequentially. Do not proceed to the next phase until the current phase is 100% verified.

### Phase 1: Infrastructure Foundation & Guardrails
*   [ ] **Verify Pinned Memory:** Run `shm_bandwidth.py`. Assert `Host -> Device` is **> 22 GB/s**.
    *   *If < 10GB/s:* You are not using Pinned Memory. Check `tensor.pin_memory()` and `ulimit`.
*   [ ] **Kernel Parity:** Verify `torch.backends.cuda.flash_sdp_enabled` is True.
*   [ ] **OS Configuration:** Run `swapoff -a` (Disable Swap) to prevent disk thrashing.
*   [ ] **NUMA Affinity:** Use `numactl --cpunodebind=X --membind=X` to bind the server process to the GPU-local CPU node.

### Phase 2: Ring Buffer Implementation (The "Moonshot")
*   [ ] **Implement Skip-End Allocator (Startup):**
    *   Code Logic: `if (offset + size > RING_SIZE) offset = 0;`
    *   Verification: Print `offset` log. Ensure no tensor starts at `RING_SIZE - 100 bytes`.
*   [ ] **Implement Pre-Computed Views:**
    *   Create a list `self.layer_views` during model load.
    *   Ensure each view points to the correct offset in the Ring Buffer.
*   [ ] **Implement Hook Swizzling:**
    *   Use `layer.register_forward_pre_hook(hook_fn)`.
    *   Inside hook: `module.weight.data = self.layer_views[layer_idx]`.
    *   **Verification:** Check `module.weight.data_ptr()` matches the ring buffer address.
*   [ ] **Disable CUDA Graphs:** Explicitly disable `torch.compile` or CUDA Graphs for the weight streaming experiment.
*   [ ] **GIL Safety Check:** Ensure the "Prefetch Loop" (Stream A) runs ahead of the "Compute Loop" (Stream B).
*   [ ] **Ablation Switch:** Implement a flag `--disable-prefetch` (forces `synchronize()` after copy) to generate the baseline data.

### Phase 3: Semantic Scheduler (The "Agent" Logic)
*   [ ] **Idle Detector:**
    *   Logic: If a Session has no active LazyTensor ops for > 1.0s, mark as `IDLE`.
    *   Action: Issue `vmu.evict_data_segment(session_id)`.
*   [ ] **Swap-to-Host:** Implement `cudaMemcpyAsync(Host, Device_KV)` for eviction.
    *   **Verification:** Watch VRAM usage drop on `nvidia-smi` when agent sleeps.
*   [ ] **Queue Fairness:** Use **LIFO** for the ReadyQueue during overload (prevents timeout for the most recent user; let the old ones wait).

### Phase 4: Metric Fidelity & Correctness
*   [ ] **Logit Equivalence Check:**
    *   Run one pass with standard PyTorch (loading model fully on CPU or 2 GPUs).
    *   Run one pass with Djinn (Ring Buffer).
    *   Pass Condition: `torch.norm(djinn_output - ref_output) < 0.1` (FP16 tolerance).
*   [ ] **The "Flatline" Test:**
    *   Run Llama-70B. Open `nvidia-smi dmon -s pcit`.
    *   Pass Condition: `rx` column should stay > 20000 MB/s consistently.
*   [ ] **The "Sleep" Check:**
    *   Start Agent. Let it generate. Sleep 30s. Resume.
    *   Pass Condition: No OOM, correct output generated.

### Phase 5: Troubleshooting "Gotchas"
*   **"My speed is 12GB/s!"** $\to$ You are likely accidentally using Pageable Memory on the host. Check `tensor.pin_memory()`.
*   **"CUDA Illegal Memory Access!"** $\to$ Your Skip-End logic failed. A tensor wrapped around the buffer end.
*   **"Latency spikes!"** $\to$ Python Garbage Collector might be pausing the Prefetch thread. Try `gc.disable()` during the inference loop.