
# Djinn: The Semantic Tensor Operating System

**Status**: Production Ready (v2.3.17)  
**Last Updated**: December 3, 2025
**Target Audience**: Systems Researchers, ML Infrastructure Engineers, Platform Architects

**Implementation Status:**
- ✅ **Phase 1 (Infrastructure):** 50% - Pinned memory verified, OS config documented, NUMA binding pending
- ✅ **Phase 2 (Ring Buffer):** 100% - Skip-End allocator, async pipelining, hook integration complete
- ✅ **Phase 3 (Semantic Scheduler):** 100% - Idle detector, swap-to-host, LIFO scheduling implemented and tested (20/20 unit tests)
- ✅ **Phase 4 (Validation):** 100% - Logit equivalence, PCIe flatline, agent sleep/resume tests implemented and validated

---

## 1. The Return of the Mainframe: GPU Clusters as the Neo-Central Utility

In the period 1965–1985, the mainframe computer represented the ultimate concentration of computing power, accessible only to the largest organizations. Today’s large GPU clusters—specifically those built on NVIDIA H100 and GB200 architectures—have recreated the economic and technical conditions of the classic mainframe era with astonishing fidelity. Consequently, the industry is repeating the evolutionary trajectory that led from primitive batch monitors to genuine operating systems.

### 1.1 Economic Equivalence: The Return of the CapEx Barrier
The primary driver of centralization is the cost of scarcity.
*   **The Mainframe Era:** A fully configured IBM System/370 Model 195 (c. 1974) cost **$38–52 million** (2025 adj.).
*   **The AI Era:** A modern NVIDIA GB200 NVL72 rack (c. 2025) commands **$52–65 million**.

The "democratization of compute" characterized by the PC era has ended for frontier workloads. We have returned to a reality where the machinery of production is centralized, necessitating a shift from personal ownership models to **Time-Sharing**.

### 1.2 The Performance Chasm
The capability gap between local and frontier hardware necessitates a "Thin Client" architecture.
*   **1975:** A CDC 7600 offered a **5,000×** speed advantage over a departmental minicomputer.
*   **2025:** A GB200 rack delivers **10,000×** the FP8 throughput of a researcher's workstation.

The modern workstation has effectively reverted to the status of a "Dumb Terminal" (or 3270 console)—an interface to submit instructions to the actual computer: the GPU cluster.

### 1.3 The Software Lag: Schedulers are Batch Monitors
Modern cluster managers (Slurm, Kubernetes) are extraordinarily capable, yet they remain architecturally aligned with the **Batch Monitors** of the 1950s (GM-NAA I/O).
*   **No Hardware Abstraction:** They allocate physical boxes ("User A gets Node 4"), not virtual resources.
*   **No Unified Memory:** They lack a facility to treat cluster VRAM as a contiguous address space.
*   **Fragility:** Node failure kills the job, exactly as in early batch systems.

**The Solution:** A new class of GPU-native operating system is not merely desirable but architecturally necessary. This OS must provide topology abstraction, unified memory addressing, and preemptive multitasking. **Djinn is that Operating System.**

---

## 2. The Design Philosophy: Why a "Semantic" OS?

To build an OS for Deep Learning, we cannot simply clone Linux. Standard Operating Systems manage *bytes*; Deep Learning requires managing *tensors*.

### The Semantic Gap
A driver-level approach (e.g., modifying CUDA) fails because it lacks semantic context. A driver sees `malloc(1GB)` but cannot distinguish between:
*   **Weights:** Read-Only, Shared, Persistent.
*   **Activations:** Volatile, Discardable, Ephemeral.
*   **KV-Cache:** Private, Stateful, Persistent.

### The Framework as the Kernel
Djinn implements the **Library OS** architecture (similar to Exokernels). It embeds itself within the application layer—specifically **PyTorch**—to bridge the gap between high-level user intent and low-level hardware execution. By intercepting execution at the framework level, Djinn acts as a **Semantic Hypervisor**, translating Python intent into optimized OS primitives.

### 2.x Theoretical Foundation

Djinn's design rests on principled separation of concerns, grounded in information theory and programming language semantics:

**Semantic Information Gap:** Framework-layer disaggregation is theoretically justified by the monotonicity of information: each layer below has strictly less semantic knowledge than the framework. A driver sees `malloc(1GB)` but a framework sees `allocate_kv_cache(1GB)`—all optimization gains come from this richer information.

**Selective Laziness:** The LazyTensor + materialization trigger model implements selective strictness in an otherwise lazy system, enabling dynamic control flow (e.g., MoE routing, conditional attention) while preserving static planning. This follows call-by-need evaluation principles, where materialization occurs only at control points, allowing planning to operate on resolved graphs.

---

## 3. System Architecture: The Big Picture

Djinn is architected like a modern Operating System, split into **User Space (Client)** and **Kernel Space (Server)**.

```
┌───────────────────────────────────────────┐      ┌───────────────────────────────────────────┐
│           USER SPACE (Client)             │      │           KERNEL SPACE (Server)           │
├───────────────────────────────────────────┤      ├───────────────────────────────────────────┤
│ 1. SYSCALL INTERFACE (LazyTensor)         │      │ 5. THE LINKER (Loader)                    │
│    • Intercepts Eager PyTorch             │      │    • Dynamic Linking (Ghost Models)       │
│    • Builds Semantically Rich Graph (SRG) │      │    • Shared Text Segment (Weights)        │
│                                           │      │                                           │
│ 2. THE COMPILER (Semantic Analysis)       │      │ 6. THE MEMORY KERNEL (Unified VMU)        │
│    • Phase Detection (Prefill vs Decode)  │      │    • Aligned Slab (Stack)                 │
│    • Optimization Hints                   │      │    • Private Heap (Data Segment)          │
│                                           │      │    • DMA Synchronization                  │
│ 3. THE I/O STACK (Serializer)             │      │                                           │
│    • Binary Protocol (No Pickle)          │      │ 7. THE SCHEDULER                          │
│    • Hybrid Transport (Coalesce/Scatter)  │      │    • Meta-Simulation (The Planner)        │
│                                           │      │    • Plan Caching (TLB)                   │
│ 4. SAFETY GUARD (Capability Engine)       │      │                                           │
│    • Local Resource Auditing              │      │ 8. GARBAGE COLLECTOR (Session Mgr)        │
└───────────────────────────────────────────┘      └───────────────────────────────────────────┘
```

---

## 4. The Frontend: From Eager Execution to Semantic Intent

Standard PyTorch is "Eager": it executes every operation immediately. This is excellent for debugging but inefficient for distributed systems.

### 4.1 The Interceptor: `LazyTensor`
Djinn introduces `LazyTensor`, a specialized tensor subclass that acts as a **Just-In-Time (JIT) Recorder**.
*   **Mechanism:** When a user runs `y = model(x)`, Djinn records the operation into a Directed Acyclic Graph (DAG) instead of executing it.
*   **Transparency:** This leverages `__torch_dispatch__`, requiring zero code changes from the user other than `model.to('remote')`.

### 4.2 The Artifact: Semantically Rich Graph (SRG)
Djinn does not just record a compute graph; it enriches the LazyTensor DAG with SRG metadata.
*   **Operation Semantics:** Distinguishes Compute-Bound ops (MatMul) from Memory-Bound ops (Attention).
*   **Phase Detection:** Automatically labels the graph as **"Prefill"** or **"Decode"**.
*   **Data Lifecycle:** Tags tensors as Ephemeral vs. Persistent, enabling the backend to optimize placement before allocation occurs.

> **Implementation note:** The SRG is a *view* over the LazyTensor DAG, not a second graph.  
> • Each node stores semantic fields lazily (phase, lifecycle, compute cost, memory hints).  
> • Subsystems (Meta-Simulator, Capability Interlock, VMU planner) query these fields directly.  
> • When we need portability or offline analysis, we materialize an SRG snapshot by walking the enriched DAG and emitting `(op, semantics, lifecycle)` tuples.  
> • This avoids duplicating graph storage while still enabling fast, query-specific caches (e.g., per-phase memory summaries) that are serialized alongside the Memory Plan.  
>
> Practically, this design keeps SRG construction overhead near-zero for common paths, yet preserves the option to export a normalized semantic graph for tooling, replay, or plan caching.

---

## 5. The Backend: The Unified Memory Kernel

To solve the fragmentation and latency issues inherent in multi-tenant GPU sharing, Djinn replaces standard allocators with an OS-inspired **Segmented Memory Model**.

### 5.1 The Unified VMU (Virtual Memory Unit)
Standard allocators (like `cudaMalloc`) fragment memory when handling concurrent dynamic workloads. Djinn's VMU partitions GPU memory into three segments, chosen to match the dominant (ownership, mutability, lifetime) classes observed in ML workloads:

| Memory Segment | OS Analogy | Lifecycle | Implementation |
| :--- | :--- | :--- | :--- |
| **Text Segment** | Shared Libs | **Read-Only** | **Model Weights.** Loaded once. Mapped into the virtual address space of every user running that model. Zero duplication. For models exceeding VRAM, uses **Ring Buffer** mode (see §5.4). |
| **Data Segment** | Heap | **Private** | **KV-Cache & Outputs.** Owned by a specific Session ID. Persists between requests to support stateful inference (e.g., Notebooks). Security is logically enforced by Session ID checks. |
| **Stack Slab** | Stack | **Volatile** | **Activations.** A massive, **256-byte aligned** scratchpad for intermediate compute. |

This segmentation minimizes duplication and maximizes cache efficiency: weights (shared, immutable) in Text; session state (private, persistent) in Data; activations (private, ephemeral) in the Stack with watermark reset, guaranteeing **zero external fragmentation** for intermediate computations.

### 5.2 The "Copy-Out" Execution Strategy
This strategy optimizes for **Zero External Fragmentation** (at the cost of minor internal fragmentation due to alignment):
1.  **Stack Allocation:** All intermediate activations use the **Stack Slab**. Allocation is a simple pointer increment (O(1)).
2.  **Execution:** The GPU computes the graph via the Slab.
3.  **Copy-Out:** Only the final requested outputs are cloned from the volatile Slab to the private **Data Segment**.
4.  **Reset:** The Slab pointer is reset to the watermark instantly. *Crucially, this reset occurs only after explicit `torch.cuda.synchronize()` ensures the GPU has finished computing.*

*Result:* A 70B model forward pass requiring 40GB of activation memory creates **zero** lasting external fragmentation. The memory is reclaimed the instant the pass completes.

### 5.3 Concurrency Model
Djinn employs a hybrid concurrency model:
*   **Space-Sharing (VRAM):** Users are isolated via private Data Segments (Heap). User A's KV Cache sits alongside User B's Weights.
*   **Time-Sharing (Compute):** The **Stack Slab** is a shared resource. Multiple users time-share the Slab for execution.
    *   **Text Segment:** Read-only, accessed concurrently by multiple CUDA streams.
    *   **Stack Slab:** Protected by stream synchronization. One user executes at a time per GPU stream, preventing data corruption.

### 5.4 Skip-End Ring Buffer (Oversized Model Support)

For models exceeding available VRAM (e.g., 140GB LLaMA-3 on a 48GB GPU), the Text Segment can be configured as a **circular ring buffer** that streams weights layer-by-layer during inference.

**Implementation:** `djinn/backend/runtime/ring_buffer.py`

```
┌─────────────────────────────────────────────────────────────┐
│ Ring Buffer (48GB)                                          │
├─────────────────────────────────────────────────────────────┤
│ [Layer 0] [Layer 1] ... [Layer N-3] [SKIP] [Layer 0] ...   │
│                                        ↑                    │
│              If next layer won't fit, skip to start         │
│              (never splits tensors across buffer wrap)      │
└─────────────────────────────────────────────────────────────┘
```

**Key Features:**
*   **Skip-End Allocation:** Pre-computes layer offsets at model registration. If a layer won't fit before the buffer end, skips to the start—tensors are never split across the wrap boundary.
*   **GPU-Resident Model Loading:** Model parameters ARE ring buffer views from initialization, eliminating device mismatch errors that plague traditional offloading approaches.
*   **Async Dual-Stream Pipelining:** High-priority prefetch stream transfers weights while compute stream executes the previous layer. Zero CPU blocking via GPU events.
*   **PyTorch Hook Integration:** `register_forward_pre_hook` transparently redirects weight pointers to ring buffer views before each layer's forward pass.
*   **Adaptive Virtualization:** Allocates as many parameters as fit in ring buffer as resident (55%), remaining parameters (45%) marked for on-demand streaming.

**Architecture (v2.3.18 - OSDI Experiment 2):**
- **Model Skeleton Loading:** Load model structure to 'meta' device (zero memory)
- **Parameter Allocation:** Allocate resident parameters to ring buffer views
- **Weight Loading:** Stream actual weights from checkpoint to GPU at 11.6 GB/s
- **Inference:** Forward pass runs entirely on GPU with resident weights, no device mismatch

**Configuration:**
```bash
export GENIE_VMU_USE_RING_BUFFER=1
export GENIE_VMU_RING_BUFFER_CAPACITY_GB=16  # For L4 (24GB GPU)
export GENIE_VMU_USE_PINNED_MEMORY=1
```

*Result:* 
- **Llama-2-13B (26GB) on L4 (24GB):** 59× faster than HuggingFace Accelerate (3.6s vs 212s)
- **Sustained H2D bandwidth:** 11.6 GB/s (pinned memory optimized)
- **Effective inference throughput:** 6.74 GB/s (compute-bound, as expected)
- **TTFT:** 72ms (vs 4,250ms for CPU offloading)
- **Memory virtualization:** 45% of parameters streamed on-demand, 55% resident

---

## 6. The Virtualization Layer: Ghost Loading & Skeletonization

Djinn decouples the *definition* of data from the *location* of data, enabling true "Time-Sharing."

### 6.1 Ghost Interception & Shadow Sync
When a user loads a model, Djinn employs one of two strategies based on the source:

*   **Strategy A: Ghost Interception (HuggingFace):** Djinn hooks `from_pretrained`. The client creates a "Ghost Model" on the `meta` device (0 bytes RAM). The server downloads the weights directly to the **Text Segment**.
*   **Strategy B: Shadow Sync (Custom Models):** For user-defined architectures (e.g., `class MyNet(nn.Module)`), Djinn computes a structural hash, serializes the weights, and uploads them to the server's Text Segment in the background. This creates a "Ghost" replica for future runs.

Djinn now streams those uploads asynchronously: each state dict is pinned on the CPU, copied over a dedicated CUDA stream, and the new `MetaSimulator` summaries (phase hint + lifecycle bytes + input bucket) are cached with the memory plan. This keeps registration efficient while still populating the segmented VMU slab with zero-copy transfers.

### 6.2 Output Skeletonization (Lazy I/O)
In interactive workflows, returning full high-dimensional tensors saturates bandwidth.
*   **The Skeleton:** Djinn returns the *structure* of the result (Dicts, Tuples) but replaces the heavy tensors with **RemoteRefStubs**.
*   **Lazy Materialization:** Data moves over the network **only** if the user explicitly accesses it (e.g., `print(logits)`).
*   **Benefit:** Reduces network bandwidth usage by **99.7%** in standard inference loops.

---

## 7. The Scheduler: Meta-Simulation, Planning & Semantic Scheduling

Traditional executors (like PyTorch Eager) calculate memory offsets at runtime, causing overhead. Djinn separates **Planning** from **Execution** and adds **Semantic Scheduling** for multi-tenant workloads.

### 7.1 The Meta-Simulator
Before execution, Djinn runs the SRG on the `meta` device (a zero-memory simulation) to calculate the exact size and lifespan of every intermediate tensor.
*   **The Output:** A deterministic **Memory Plan** mapping every operation to an exact offset in the **Stack Slab**.
*   **Benefit:** Eliminates runtime `malloc/free` calls entirely.
*   **Summary:** Every plan caches a lightweight SRG summary (`phase_hint`, `lifecycle` breakdowns, input bucket) so telemetry tools can ingest it without traversing the LazyTensor DAG again.

### 7.2 Plan Caching (The "TLB")
Simulating the graph takes time (~50ms). To achieve sub-millisecond latency, Djinn caches Plans.
*   **The Cache Key:** `(Model_Fingerprint, Input_Shape_Tuple)`.
*   **The TLB Effect:** For repeated requests (e.g., generating tokens 2, 3, 4...), the Scheduler skips simulation and loads the offsets from the cache (O(1) lookup).
*   **Result:** Reduces scheduling latency from 50ms to **<0.5ms**.

### 7.3 Basic QoS Classes (Realtime / Interactive / Batch)
Multi-tenant performance collapses when every request fights for the same slot. Djinn now exposes **three QoS classes** that can be selected explicitly (`hints={'qos_class': 'realtime'}`) or inferred from deadlines:

*   **Realtime:** Strict priority, capped latency, intended for token-by-token decoding or streaming speech. Reserved concurrency slices ensure a realtime request is never starved by batch uploads.
*   **Interactive:** Default class for chatbots, notebook users, or dashboards. Shares the bulk of concurrency slots and is protected from background drains.
*   **Batch:** Background work (offline evals, artifact builds). Runs opportunistically and yields whenever higher classes arrive.

Under the hood a **Basic QoS Scheduler** keeps per-class queues, enforces configurable concurrency shares, and records per-request queue latency so we can chart SLA compliance. The scheduler is on by default (`GENIE_ENABLE_QOS=1`) but can be tuned via `GENIE_QOS_MAX_CONCURRENCY` and `GENIE_QOS_CLASS_SHARES` for different fleet profiles.

### 7.4 Semantic Scheduler: Intelligent KV Cache Management (Phase 3)

For multi-agent and long-context workloads, the **Semantic Scheduler** proactively manages KV cache eviction by understanding application-level semantics rather than relying on reactive heuristics.

**Three Components:**

1. **Idle Detector (SemanticActivityTracker)**
   - Monitors session activity (model execution timestamps)
   - Marks sessions idle after configurable threshold (default: 1.0s)
   - Asynchronously notifies KV manager via event loop integration

2. **Swap-to-Host (HostSwapPool + KVSessionManager)**
   - Pre-allocated pinned CPU memory pool for KV cache eviction
   - On idle detection: `cudaMemcpyAsync` transfers KV to host (frees GPU VRAM)
   - On resume: Restores KV from host back to GPU via dedicated transfer stream
   - Zero GPU synchronization overhead (stream-specific, not global `torch.cuda.synchronize()`)

3. **Queue Fairness (LIFO Scheduling)**
   - During system overload (queue depth > 2× concurrency): switches to LIFO pop
   - Ensures newly-arriving requests don't timeout waiting for old ones
   - Metrics: `lifo_switches` counter tracks overload transitions

**Configuration:**
```bash
python -m djinn.server.server_main \
    --enable-semantic-scheduler \
    --idle-threshold-seconds 1.0 \
    --host-swap-pool-gb 32
```

**Benefit:** Enables 50+ concurrent agents with 2GB KV cache each (100GB total) on 60GB GPU by proactively swapping idle sessions, vs. vLLM's reactive OOM.

---

## 8. The I/O Subsystem: High-Performance Plumbing

Connecting the Client and Server is a custom networking stack designed for Tensor workloads, addressing the "Serialization Bottleneck."

### 8.1 DjinnSerializer (Binary Protocol)
We replace Python's `pickle` (3-4ms overhead) with a **Length-Prefixed Binary Protocol**.
*   **Structure:** JSON header for metadata + Raw Byte Append for data.
*   **Zero-Copy:** Tensors are read directly from memory using `memoryview`, bypassing user-space copies.
*   **Performance:** Reduces serialization latency to **<0.5ms**.

### 8.2 Hybrid Transport & DMA
*   **Hybrid Coalescing:** Small requests (<1400B, e.g., prompts) are packed into single packets to minimize syscalls.
*   **Synchronized DMA:** On the server, data flows from Network → Pinned CPU Staging → GPU Slab via a **synchronized DMA pipeline**, ensuring the GPU never reads corrupted data during async transfers.

---

## 9. Reliability: The Kernel Guard & Validation

An OS must be robust and verifiable. Djinn implements safeguards and comprehensive validation tests.

### 9.1 Runtime Safeguards

*   **Capability Interlock (Client-Side OOM Killer):** Before falling back to local execution (if the cluster is busy), Djinn audits local RAM. If the machine lacks resources (requires 1.5x headroom), it halts execution with a `ResourceError` instead of freezing the host OS.
*   **Session GC (Distributed Garbage Collection):** Distributed memory leaks are fatal. Djinn tracks **Session Leases** monitored by heartbeats. If a client disconnects, the Server immediately reclaims their private **Data Segment**, ensuring zero VRAM leaks.

### 9.2 Phase 4: Metric Fidelity & Correctness Validation

Three core validation tests verify system correctness and performance claims:

1. **Logit Equivalence Check**
   - Compares Djinn (with Ring Buffer) vs standard PyTorch baseline
   - Pass criterion: `torch.norm(djinn_output - pytorch_output) < 0.1` (FP16 tolerance)
   - Verifies Ring Buffer's lazy weight streaming doesn't corrupt computation

2. **PCIe Flatline Test** 
   - Runs Llama-70B inference with continuous PCIe monitoring
   - Pass criterion: PCIe RX bandwidth > 20000 MB/s sustained (>80% of time)
   - Validates Async Dual-Stream Pipelining hides prefetch latency behind compute

3. **Agent Sleep/Resume Test**
   - Simulates 10s idle agent with KV eviction + restoration
   - Pass criterion: No OOM, coherent text generation before/after sleep
   - Validates Semantic Scheduler's swap-to-host mechanism end-to-end

**Implementation:** `OSDI_Evaluation/phase4_validation/`

---

## 10. Performance Summary

By moving from a "Network Wrapper" to a "Tensor Operating System," Djinn achieves performance metrics that validate the approach:

| Metric | Baseline (Graph-Based) | Djinn v2.3 (Tensor OS) | Improvement |
| :--- | :--- | :--- | :--- |
| **Latency (Small)** | 31ms | **0.8ms** | **38x Faster** (via Plan Caching) |
| **Latency (Large)** | 868ms | **81ms** | **10.7x Faster** (via Zero-Copy) |
| **Bandwidth** | 100% (Full Return) | **0.3%** | **99.7% Reduction** (via Skeletonization) |
| **Fragmentation** | High (Standard Allocator) | **Zero (External)** | **Unified VMU (Slab)** |

### Experiment 2: Memory Virtualization (v2.3.18 - OSDI Evaluation)

**Ring Buffer Virtualization Results (L4 GPU, Llama-2-13B):**

| Metric | HF Accelerate | Djinn Ring Buffer | **Speedup** |
|--------|---------------|-------------------|-------------|
| **Latency** | 212,517 ms | 3,599 ms | **59× Faster** |
| **Bandwidth** | 0.11 GB/s | 6.74 GB/s | **61× Higher** |
| **TTFT** | 4,250 ms | 72 ms | **59× Faster** |
| **Peak VRAM** | 5.7 GB | 16.2 GB | - |

**Key Achievement:** GPU-resident model loading with ring buffer virtualization delivers **59× speedup** over standard HuggingFace Accelerate CPU offloading. Enables running 26GB models on 24GB GPUs with 45% parameter virtualization and 99.7% ring buffer utilization, eliminating device mismatch errors inherent to traditional offloading.

---

## Contact & Citation

**Project Lead**: Jaewan Hong (jaewan@berkeley.edu)  

```bibtex
@inproceedings{hong2025lost,
  title={Lost in Translation: The Search for Meaning in Network-Attached AI Accelerator Disaggregation},
  author={Hong, Jaewan and Qiao, Yifan and Ponnapalli, Soujanya and Liu, Shu and Aguilera, Marcos K and Liu, Vincent and Rossbach, Christopher J and Stoica, Ion},
  booktitle={Proceedings of the 24th ACM Workshop on Hot Topics in Networks},
  pages={131--138},
  year={2025}
}
```
