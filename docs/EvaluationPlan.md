Wear OSDI reviewer's hat and evaluate our evaluation plan and your suggestion with 20+ years of experience. Isn't it too complex with too many evaluations?
What are the most essential experiments and how should we run them? given our OVERVIEW.md what baselines should we run?

After thinking about these, assess my revised evaluation plan below.

# Djinn Evaluation Plan (Final Revision)
**Version:** 3.0 - Submission Ready  
**Timeline:** 8 weeks (6 weeks experiments + 2 weeks buffer/writing)  
**Scope:** 6 pages, 8-10 figures/tables, 4-5 focused experiments

---

## Executive Summary

This evaluation plan proves Djinn's core thesis through **5 focused experiments** that demonstrate:

1. **Semantic awareness transforms disaggregation** (10-100× improvement)
2. **System scales efficiently** with multi-tenant QoS guarantees
3. **Memory kernel works** with OS-level properties
4. **Overhead is acceptable** (<5% for production models)
5. **Design generalizes** across workload families

**Key Innovation:** All baselines are Djinn variants with features disabled, ensuring fair comparison.

---

## Core Evaluation Questions

| Question | Experiment | Figures | Critical? |
|----------|------------|---------|-----------|
| Does semantic awareness matter? | §6.2 Semantic Benefit | 2-3 | ✅ YES |
| Which components contribute? | §6.3 System Analysis | 3-4 | ✅ YES |
| Does it scale with QoS? | §6.4 Multi-Tenant | 2-3 | ✅ YES |
| What's the overhead? | §6.5 Overhead | 1-2 | ✅ YES |
| Does it generalize? | §6.6 Generality | 1-2 | ⚠️ NICE |

---

## Baseline Strategy (Critical Design Decision)

### The Four Baselines

All baselines use **Djinn infrastructure** with different configurations:

```python
BASELINES = {
    "Native PyTorch": {
        # Upper bound: local execution
        "location": "local_gpu",
        "semantic_features": "N/A"
    },
    
    "Semantic-Blind Djinn": {
        # Djinn with ALL semantic features disabled
        "phase_detection": False,
        "lifecycle_tracking": False,
        "stateful_caching": False,
        "plan_cache": False,
        "skeletonization": False,
        # KEEP: binary_protocol=True, vmu=True (not semantic)
    },
    
    "Partially-Aware Djinn": {
        # Djinn with SOME semantic features
        "phase_detection": False,  # Blind to phases
        "lifecycle_tracking": False,  # Blind to lifecycles
        "stateful_caching": True,  # But model caching works
        "plan_cache": True,
        "skeletonization": True,
    },
    
    "Full Djinn": {
        # All features enabled
        "phase_detection": True,
        "lifecycle_tracking": True,
        "stateful_caching": True,
        "plan_cache": True,
        "skeletonization": True,
    }
}
```

### Why This Is Fair

✅ **Same hardware:** All run on same A100-80GB GPU  
✅ **Same infrastructure:** All use Djinn's networking, VMU, serialization  
✅ **Same models:** Identical model weights and configurations  
✅ **Only difference:** Semantic awareness enabled/disabled  
✅ **Implementable:** Config flags, not separate systems (1 week work)

**Critical:** Semantic-blind still uses binary protocol and VMU because these are **not semantic features** (they're performance optimizations independent of semantics). This prevents creating a "crippled strawman."

---

## Section 6.2: Semantic Awareness Benefit [Hero Experiment #1]

**Goal:** Prove semantic awareness transforms disaggregation efficiency

### Experiment 2.1: LLM Decode with KV Cache

**The Killer Experiment** - If you do ONE experiment well, do this one.

**Setup:**
```python
Model: LLaMA-7B (7B parameters, 12GB weights)
Workload: Generate 50 tokens from 72-token prompt
Hardware: Client (CPU) → Server (A100-80GB) via 25 Gbps network

Baselines: All 4 (Native, Blind, Partial, Full)

Metrics (per-token averages):
- Latency (ms)
- Data transferred (MB)
- GPU utilization (%)
- Time breakdown (compute vs transfer vs queue)
```

**Expected Results:**

| Baseline | Latency/token | Data TX (50 tok) | GPU Util | Why? |
|----------|---------------|------------------|----------|------|
| **Native PyTorch** | 3ms | 0 GB | 99% | Upper bound (local) |
| **Semantic-Blind** | 850ms | 60 GB | 3% | Sends full model state each token |
| **Partially-Aware** | 110ms | 4 GB | 18% | Caches model but not KV cache |
| **Full Djinn** | 15ms | 0.8 MB | 85% | KV cache co-located, handle-only |

**Key Insights:**
- Semantic-blind is **57× slower** than Full Djinn
- Data transfer reduction: **75,000×** (60 GB → 0.8 MB)
- GPU utilization improvement: **28×** (3% → 85%)

**Deliverables:**
- **Figure 6a:** Bar chart (4 baselines × latency/throughput/data)
- **Figure 6b:** Time breakdown stacked bars (compute / transfer / queue / idle)
- **Figure 6c:** GPU utilization over 50 token generation (line plot)

**Why This Proves Thesis:**
This single experiment demonstrates:
1. Semantic blindness causes catastrophic performance (850ms vs 15ms)
2. Progressive improvement from partial → full awareness
3. Root cause: data movement dominates when semantics are lost
4. Djinn's phase detection + stateful co-location solves this

---

### Experiment 2.2: Streaming Audio Transcription (Djinn's Sweet Spot)

**Setup:**
```python
Model: Whisper-Large (1.5B parameters, 3GB weights)
Workload: Transcribe 30-second audio clip with sliding window (5s chunks)
Hardware: Client (CPU) → Server (A100-80GB) via 25 Gbps network

Semantic optimization: State persistence across chunks, skeletonization

Baselines: All 4 (Native, Blind, Partial, Full)
```

**Expected Results:**

| Baseline | Latency | Data TX | GPU Util | Why? |
|----------|---------|---------|----------|------|
| **Native PyTorch** | 120ms | 0 GB | 95% | Local upper bound |
| **Semantic-Blind** | 850ms | 45 GB | 8% | Sends full model each chunk |
| **Partially-Aware** | 180ms | 2.3 GB | 45% | Caches model, sends state |
| **Full Djinn** | 35ms | 12 MB | 82% | Persistent state + skeletonization |

**Key Insights:**
- **34× faster** than semantic-blind (850ms → 35ms)
- **Data transfer reduction: 3,750×** (45GB → 12MB)
- **GPU utilization: 10× improvement** (8% → 82%)

**Why This Showcases Djinn:**
- **Sequential processing** (perfect for Djinn's strengths)
- **Stateful workload** (KV cache persistence across chunks)
- **Real-time requirements** (low latency critical)
- **Data amplification** (semantic blindness causes massive transfer overhead)

**Deliverables:**
- **Figure 7a:** Latency comparison across baselines
- **Figure 7b:** Data transfer vs. processing time breakdown
- **Figure 7c:** GPU utilization over streaming window

### Experiment 2.3: Conversational AI (Multi-Turn Dialogue)

**Setup:**
```python
Model: GPT-J-6B (6B parameters, 24GB weights)
Workload: 10-turn conversation (prompt + 9 responses)
Hardware: Client (CPU) → Server (A100-80GB) via 25 Gbps network

Semantic optimization: KV cache evolution, session state management

Baselines: Semantic-Blind vs Full Djinn
```

**Expected Results:**
- **Semantic-Blind:** 45 GB data transfer, 12s total latency, 15% GPU util
- **Full Djinn:** 280 MB data transfer, 890ms total latency, 78% GPU util
- **Improvement:** 13× faster, 160× less data transfer

**Why This Matters:**
- Real conversational workloads (not synthetic benchmarks)
- Demonstrates Djinn's strength in stateful, sequential processing
- Shows progressive KV cache growth optimization

**Deliverables:**
- **Figure 8:** Conversation flow efficiency (latency per turn)

---

## Section 6.3: System Analysis [Hero Experiments #2 & #3]

### Experiment 3.1: Component Ablation Study

**Goal:** Identify which components contribute how much

**Setup:**
```python
# Start with Full Djinn, remove one component at a time

configurations = {
    "Full Djinn": {ALL_ENABLED},
    "No VMU": {USE_CUDAMALLOC},  # Critical: shows memory kernel value
    "No Phase Detection": {BLIND_TO_PHASES},
    "No Stateful Caching": {NO_KV_PERSISTENCE},
    "No Plan Cache": {RECOMPUTE_PLANS},
    "No Skeletonization": {RETURN_FULL_TENSORS},
}

# Test on 2 workloads
workloads = ["llama-7b-generation", "resnet50-batch"]

# Measure: throughput, latency, memory efficiency
```

**Expected Results (LLM workload):**

| Configuration | Throughput | Latency | Memory Util | Impact |
|---------------|------------|---------|-------------|--------|
| **Full Djinn** | 100% | 15ms | 91% | Baseline |
| No Skeletonization | 18% | 78ms | 91% | -82% (data flood) |
| No Stateful Caching | 35% | 42ms | 91% | -65% (KV retransfer) |
| No Phase Detection | 48% | 28ms | 91% | -52% (blind scheduling) |
| No VMU | 65% | 21ms | 68% | -35% (fragmentation) |
| No Plan Cache | 87% | 17ms | 91% | -13% (planning overhead) |

**Component Ranking:**
1. **Skeletonization** (82% contribution) - avoids output data flood
2. **Stateful Caching** (65% contribution) - KV cache co-location
3. **Phase Detection** (52% contribution) - prefill/decode optimization
4. **VMU** (35% contribution) - memory efficiency
5. **Plan Cache** (13% contribution) - planning overhead

**Deliverables:**
- **Figure 8a:** Ablation results (stacked bar chart showing component impact)
- **Figure 8b:** Memory utilization (VMU vs cudaMalloc over 1-hour run)

**Critical Addition (Reviewer Feedback):**
**Figure 8b** explicitly shows VMU's OS-level memory properties:
- **Zero external fragmentation** (VMU flat at 91%, cudaMalloc degrades to 68%)
- **Stable allocation latency** (VMU O(1), cudaMalloc O(log n))
- **Higher effective utilization** under multi-tenant load

---

### Experiment 3.2: Memory Kernel Deep Dive

**Goal:** Demonstrate VMU provides OS-level memory guarantees through Djinn-specific workloads

**Setup:**
```python
# Djinn-specific memory stress test: Concurrent stateful sessions

# Create 8 concurrent "conversation sessions" (each with evolving KV cache)
# Each session: 100 allocations/deallocations of varying sizes (256B-2GB)
# Pattern: Allocate → Use → Free → Reallocate (simulates Djinn's copy-out strategy)

allocators = ["VMU", "cudaMalloc", "PyTorch_caching"]

# Measure every 30 seconds during 2-hour test:
# - External fragmentation: 1 - (used / allocated)
# - Peak memory utilization under concurrent load
# - Allocation latency (P50, P99, P99.9)
# - Memory access patterns (sequential vs random)
# - Session isolation (no cross-contamination)
```

**Expected Results:**

| Allocator | Frag (Peak) | Peak Util | Alloc P99.9 | Session Isolation |
|-----------|-------------|-----------|-------------|-------------------|
| VMU | 0% | 91% | 15 µs | ✅ Perfect |
| cudaMalloc | 28% | 68% | 450 µs | ❌ Contamination |
| PyTorch | 15% | 71% | 280 µs | ⚠️ Partial |

**Key Finding:** VMU maintains **zero external fragmentation** under realistic Djinn workloads while providing **perfect session isolation** and **sub-20µs allocation latency**.

**Djinn-Specific Validation:**
- **Session Isolation:** Each conversation's KV cache stays separate
- **Copy-Out Efficiency:** Frequent allocate/use/free cycles don't fragment
- **Concurrent Safety:** Multiple sessions don't interfere
- **Stateful Persistence:** Memory remains stable across session lifetime

**Deliverables:**
- **Figure 9a:** Fragmentation under concurrent sessions (shows VMU stability)
- **Figure 9b:** Allocation latency CDF (Djinn's O(1) allocation)
- **Figure 9c:** Session isolation demonstration (memory boundaries preserved)
- **Table 3:** Memory kernel properties under Djinn workloads

**Why This Matters:** Validates Djinn's OS kernel claims with evidence that matters:
- Zero fragmentation under **realistic workloads** (not synthetic)
- Perfect isolation (security/correctness guarantee)
- Sub-20µs allocation (enables fine-grained memory management)
- Stable under concurrent stateful sessions (production requirement)

---

## Section 6.4: Multi-Tenant Scalability with QoS [Hero Experiment #2]

### Experiment 4.1: Scalability Under Load

**Goal:** Prove system scales efficiently to realistic multi-tenant loads

**Setup:**
```python
# Single A100-80GB GPU
# Workload: LLaMA-7B mixed prefill/decode

concurrent_users = [1, 2, 4, 8, 16, 32, 64]

# Poisson arrivals: 1 req/sec per user
# Run for 300 seconds (5 minutes) per configuration

Metrics:
- Aggregate throughput (tokens/sec)
- Per-user latency P50, P99
- GPU utilization (SM occupancy)
- Memory bandwidth utilization
```

**Expected Results:**

| Users | Throughput | P99 Latency | GPU Util | Efficiency |
|-------|------------|-------------|----------|------------|
| 1 | 66 tok/s | 15ms | 52% | 100% |
| 4 | 240 tok/s | 18ms | 73% | 91% |
| 8 | 450 tok/s | 22ms | 82% | 85% |
| 16 | 780 tok/s | 28ms | 85% | 74% |
| 32 | 1,020 tok/s | 42ms | 86% | 48% |
| 64 | 1,100 tok/s | 98ms | 86% | 26% |

**Bottleneck Analysis:**
- **1-16 users:** Linear scaling (compute-bound, 74-100% efficiency)
- **16-32 users:** Memory bandwidth saturation (~1.8 TB/s)
- **32-64 users:** Scheduler CPU-bound (queue delays dominate)

**Deliverables:**
- **Figure 10a:** Throughput vs concurrent users (with linear reference line)
- **Figure 10b:** P99 latency vs users (flat until saturation)
- **Figure 10c:** GPU utilization vs users (saturates at 85%)

---

### Experiment 4.2: QoS Guarantees Under Contention

**Goal:** Demonstrate framework-semantic scheduling enables QoS differentiation

**Setup:**
```python
# Mixed workload on single A100-80GB

workload_mix = {
    "Realtime": {
        "percentage": 20%,
        "target_p99": 10ms,
        "priority": HIGH,
        "examples": "interactive_chat"
    },
    "Interactive": {
        "percentage": 30%,
        "target_p99": 100ms,
        "priority": MEDIUM,
        "examples": "web_api_calls"
    },
    "Batch": {
        "percentage": 50%,
        "target_p99": None,
        "priority": LOW,
        "examples": "document_processing"
    }
}

# Compare:
# 1. FCFS (no QoS, baseline)
# 2. Basic Priority (Djinn's current implementation)

# Run for 1 hour, measure:
# - Per-class P99 latency
# - SLA violation rate
# - Queue wait time distribution
# - GPU utilization
```

**Expected Results:**

| Class | Target P99 | FCFS P99 | Priority P99 | FCFS Violations | Priority Violations |
|-------|------------|----------|--------------|-----------------|---------------------|
| Realtime | 10ms | 87ms | 9.2ms | 95% | 4% |
| Interactive | 100ms | 240ms | 82ms | 78% | 0% |
| Batch | None | 380ms | 420ms | N/A | N/A |

**Key Finding:** Priority scheduling with semantic phase detection reduces SLA violations from 78-95% to 0-4%.

**Deliverables:**
- **Figure 11a:** Per-class latency CDF (FCFS vs Priority)
- **Figure 11b:** SLA violation rate over time (shows stability)
- **Table 4:** Queue behavior breakdown (wait time by class)

**Why This Matters:** Ties directly to "Tensor OS" / mainframe narrative:
- Time-sharing GPU resource (OS property)
- QoS guarantees via semantic scheduling
- Batch doesn't starve realtime (fair allocation)

---

## Section 6.5: Overhead Analysis

### Experiment 5.1: Framework-Layer Overhead (Djinn's Value Proposition)

**Goal:** Demonstrate Djinn's semantic optimizations justify the framework overhead

**Setup:**
```python
# Compare Djinn vs Semantic-Blind (not Native PyTorch)

model_sizes = [
    ("BERT-Base", 110M),
    ("GPT-2-Small", 117M),
    ("BERT-Large", 340M),
    ("GPT-J", 6B),
    ("LLaMA-7B", 7B),
    ("LLaMA-13B", 13B)
]

for model, params in model_sizes:
    # Measure both latency AND data transfer
    blind_results = benchmark_semantic_blind(model, n=100)
    djinn_results = benchmark_djinn(model, n=100, cache_warm=True)

    overhead_pct = (djinn_results.latency - blind_results.latency) / blind_results.latency * 100
    data_savings = (blind_results.data_tx - djinn_results.data_tx) / blind_results.data_tx * 100

    # Break down overhead sources via profiling
    overhead_breakdown = profile_djinn_overhead(model)
```

**Expected Results:**

| Model | Params | Blind Latency | Djinn Latency | Overhead | Data Savings | Net Benefit |
|-------|--------|---------------|---------------|----------|--------------|-------------|
| BERT-Base | 110M | 45ms | 48ms | +6.7% | 85% | 13× effective |
| GPT-2-Small | 117M | 52ms | 55ms | +5.8% | 82% | 11× effective |
| BERT-Large | 340M | 120ms | 125ms | +4.2% | 91% | 22× effective |
| GPT-J | 6B | 850ms | 185ms | -78% | 99.7% | 4.6× faster |
| LLaMA-7B | 7B | 1,200ms | 214ms | -82% | 99.8% | 5.6× faster |
| LLaMA-13B | 13B | 2,100ms | 384ms | -82% | 99.8% | 5.5× faster |

**Analysis:**
- **Framework overhead is minimal** (<7% for small models)
- **For large models, Djinn is NET FASTER** (-78% to -82% latency)
- **Data transfer savings** (82-99.8%) dwarf overhead costs
- **Breakdown:** Graph construction (1-3%), Network protocol (3-4%), Execution (0.1-1%)

**Djinn's Value Proposition:**
- Small models: <7% overhead for massive data savings
- Large models: **Negative overhead** (net performance gain)
- Semantic awareness pays for itself through reduced data movement

**Deliverables:**
- **Figure 12a:** Overhead vs model size (shows diminishing overhead)
- **Figure 12b:** Data savings vs overhead (benefit-cost analysis)
- **Figure 12c:** Net performance impact (latency improvement considering overhead)

**Success Criteria:**
- Overhead <10% for all models
- Net performance benefit for production-scale models (>1B params)
- Clear benefit-cost justification for framework layer

---

## Section 6.6: Cross-Workload Generality

### Experiment 6.1: Diverse Model Families (Semantic Awareness Breadth)

**Goal:** Demonstrate Djinn's semantic abstractions generalize across workload types

**Setup:**
```python
# Focus on workloads where semantic awareness matters most
model_suite = {
    "Sequential/Stateful": {
        "llama-7b": "KV cache evolution (decode phase)",
        "whisper-large": "Audio chunk streaming (temporal state)",
        "gpt-j-6b": "Conversation continuity (session state)"
    },
    "Parallel/Stateless": {
        "resnet50": "Layer pipeline parallelism",
        "vit-base": "Attention batching",
        "efficientnet": "Stage-aware placement"
    },
    "Hybrid/Stateful": {
        "clip-vit-l": "Joint embedding optimization",
        "blip": "Cross-modal state management",
        "speech-encoder": "Streaming feature extraction"
    }
}

# Djinn-specific metrics:
# - Semantic efficiency ratio: (data_saved) / (framework_overhead)
# - State persistence benefit: latency_without_state / latency_with_state
# - Lifecycle optimization: phases_detected × state_transitions_optimized
```

**Expected Results:**

| Category | Model | Speedup | Data Savings | Semantic Efficiency | Key Optimization |
|----------|-------|---------|--------------|---------------------|------------------|
| **Sequential** | LLaMA-7B | 57× | 99.8% | 180× | KV cache co-location |
| **Sequential** | Whisper-Large | 34× | 99.7% | 120× | Streaming state persistence |
| **Sequential** | GPT-J-6B | 48× | 99.8% | 160× | Session state management |
| **Parallel** | ResNet-50 | 4× | 85% | 12× | Layer pipelining |
| **Parallel** | ViT-Base | 3.2× | 78% | 8× | Attention batching |
| **Hybrid** | CLIP | 6× | 92% | 15× | Modality-aware placement |

**Djinn-Specific Analysis:**
- **Semantic Efficiency Ratio:** How much data savings per unit framework overhead
- **Sequential workloads:** 120-180× efficiency (perfect for Djinn)
- **Parallel workloads:** 8-15× efficiency (still valuable)
- **100% execution success** with semantic-aware optimizations

**Why This Matters:**
- Shows Djinn excels at **stateful, sequential processing** (its sweet spot)
- Demonstrates **progressive benefit** even for parallel workloads
- Validates semantic abstractions work across **diverse model architectures**

**Deliverables:**
- **Figure 13a:** Speedup by workload category (highlights sequential advantage)
- **Figure 13b:** Semantic efficiency ratio (Djinn's value metric)
- **Figure 13c:** Data savings vs. speedup correlation
- **Table 5:** Model coverage with Djinn-specific optimizations

---

## Timeline & Resource Plan

### Week-by-Week Breakdown

**Week 0: Preparation (Pre-work)**
```
□ Implement 4 baseline configurations
□ Build experiment harness with metrics collection
□ Validate measurements are correct
□ Set up automated plotting scripts

Deliverable: Infrastructure ready, baselines verified
Team: 2 people, 20 hours each
```

**Week 1-2: Core Experiments**
```
□ Experiment 2.1: LLM decode (THE critical one)
  - Run all 4 baselines × 30 runs
  - Generate Figure 6a-c

□ Experiment 2.2: Streaming audio transcription
  - Run all 4 baselines × 30 runs
  - Generate Figure 7a-c

□ Experiment 2.3: Conversational AI
  - Run 2 baselines × 30 runs
  - Generate Figure 8

□ Statistical analysis and sanity checks

Deliverable: Figures 6-8, core thesis proven + Djinn sweet spots
Team: 2 people, 45 hours each
```

**Week 3: System Analysis**
```
□ Experiment 3.1: Ablation study
  - Run 6 configurations × 2 workloads × 30 runs
  - Generate Figure 8a-b
  
□ Experiment 3.2: Memory kernel deep dive
  - 24-hour stress test (run over weekend)
  - Generate Figure 9a-b, Table 3

Deliverable: Figures 8-9, Table 3, component contributions
Team: 2 people, 35 hours each
```

**Week 4: Scalability & QoS**
```
□ Experiment 4.1: Scalability
  - Run 7 user counts × 30 runs
  - Generate Figure 10a-c
  
□ Experiment 4.2: QoS guarantees
  - 1-hour mixed workload × 5 runs
  - Generate Figure 11a-b, Table 4

Deliverable: Figures 10-11, Table 4, scalability proven
Team: 2 people, 35 hours each
```

**Week 5: Overhead & Generality**
```
□ Experiment 5.1: Overhead analysis
  - Run 5 model sizes × 100 runs each
  - Generate Figure 12
  
□ Experiment 6.1: Cross-workload
  - Run 6 models × 30 runs
  - Generate Figure 13, Table 5

Deliverable: Figures 12-13, Table 5, generality proven
Team: 2 people, 30 hours each
```

**Week 6: Analysis & Draft**
```
□ Generate all final figures (publication quality)
□ Run statistical significance tests
□ Write first draft of evaluation section
□ Internal review

Deliverable: Evaluation section draft (6 pages)
Team: 3 people, 25 hours each
```

**Week 7-8: Buffer & Polish**
```
□ Address internal feedback
□ Re-run any questionable experiments
□ Polish figures and writing
□ External colleague review
□ Final revisions

Deliverable: Camera-ready evaluation section
Team: 3 people, 20 hours each
```

### Total Resource Requirements

**Time:**
- Core experiments: 6 weeks
- Buffer/polish: 2 weeks
- **Total: 8 weeks calendar time**

**Compute:**
- GPU hours: ~200 hours (4 weeks of continuous running)
- Can parallelize with 2 GPUs: 2 weeks wall-clock

**Human Effort:**
- Person-weeks: ~12 person-weeks over 8 weeks
- With 2-3 people: achievable

---

## Risk Management

### High-Priority Risks

**Risk 1: Semantic-blind performance worse than expected**
- **Impact:** Makes comparison less dramatic
- **Mitigation:** Pre-test semantic-blind config in Week 0
- **Contingency:** If only 10× improvement (not 50×), it's still strong

**Risk 2: Scalability saturates earlier than hoped**
- **Impact:** Might saturate at 16 users, not 32
- **Mitigation:** This is actually fine—honest reporting is better
- **Contingency:** Focus on identifying bottleneck clearly

**Risk 3: VMU benefits not visible in 24-hour test**
- **Impact:** Memory kernel claim weakened
- **Mitigation:** Run synthetic stress test to induce fragmentation
- **Contingency:** Use theoretical analysis + micro-benchmarks

**Risk 4: One experiment fails to complete**
- **Impact:** Missing results
- **Mitigation:** 2-week buffer in timeline
- **Contingency:** Generality (Exp 6.1) is nice-to-have, can be cut

### Schedule Checkpoints

**Week 2 Decision Point:**
- If Experiment 2.1 (LLM decode) looks weak → investigate deeply
- This is THE critical experiment; don't proceed without strong results

**Week 4 Decision Point:**
- If on schedule → proceed to Week 5-6
- If behind → cut Experiment 6.1 (generality), focus on core

**Week 6 Decision Point:**
- If quality sufficient → proceed to polish
- If needs more work → use Week 7-8 buffer for additional experiments

---

## Deliverables Summary

### Figures (8-10 total)

**Core Thesis (Figures 6-7):**
- Figure 6a-c: Semantic awareness benefit (LLM decode)
- Figure 7: Vision pipeline comparison

**System Analysis (Figures 8-9):**
- Figure 8a: Ablation study
- Figure 8b: VMU vs cudaMalloc memory behavior
- Figure 9a-b: Memory kernel deep dive

**Scalability (Figures 10-11):**
- Figure 10a-c: Multi-tenant scalability
- Figure 11a-b: QoS guarantees

**Supporting (Figures 12-13):**
- Figure 12: Overhead analysis
- Figure 13: Cross-workload generality

### Tables (3-5 total)

- Table 3: Memory kernel properties
- Table 4: QoS queue behavior
- Table 5: Model coverage matrix

---

## Quality Standards

### Statistical Rigor

**Required for all experiments:**
- ✅ N ≥ 30 runs per configuration
- ✅ Report mean ± 95% confidence interval
- ✅ Report P50, P95, P99 for latencies
- ✅ Statistical significance tests (t-test or Mann-Whitney)
- ✅ Check for outliers and explain

### Figure Quality

**Every figure must have:**
- ✅ Clear title describing what is measured
- ✅ Axis labels with units
- ✅ Legend when comparing multiple series
- ✅ Error bars (95% CI)
- ✅ Annotations highlighting key insights
- ✅ 2-3 sentence caption explaining main takeaway

### Reproducibility

**Artifact package must include:**
- ✅ All baseline configurations (YAML files)
- ✅ Experiment runner scripts
- ✅ Raw data from our runs
- ✅ Plotting scripts to regenerate all figures
- ✅ Statistical analysis code
- ✅ Docker container for environment
- ✅ README with setup instructions

---

## Success Criteria

### Djinn-Specific Success Criteria (Must-Have for OSDI Acceptance)

1. ✅ **Semantic awareness transforms disaggregation** (30-57× improvement on sequential workloads)
2. ✅ **Framework overhead justified by benefits** (net performance gain for production models)
3. ✅ **VMU provides OS-level memory guarantees** (0% fragmentation, perfect isolation)
4. ✅ **Stateful workloads excel** (KV cache, streaming, conversations)
5. ✅ **Semantic efficiency ratio >10×** for target workloads

### Workload-Specific Targets

| Workload Type | Speedup Target | Data Savings | GPU Util Improvement |
|---------------|----------------|--------------|---------------------|
| **LLM Decode** | 50× | 99.8% | 20× |
| **Audio Streaming** | 30× | 99.7% | 10× |
| **Conversational** | 40× | 99.5% | 15× |
| **Vision Pipeline** | 3-4× | 80% | 2× |

### Nice-to-Have for Strong Accept

1. ⭐ **QoS scheduling works** (if implemented: SLA adherence under contention)
2. ⭐ **Broad generality** (works across 6+ model families)
3. ⭐ **Bottleneck characterization** (memory bandwidth at 16-32 users)
4. ⭐ **Real-world workloads** (conversational AI, streaming ASR)

---

## Alignment with OVERVIEW.md Claims

| Claim in OVERVIEW.md | Proven By |
|----------------------|-----------|
| "Semantic translation gap causes inefficiency" | Experiment 2.1 (57× difference) |
| "Framework layer is narrow waist" | Experiment 2.1 (progressive benefit) |
| "Tensor OS with memory kernel" | Experiment 3.2 (VMU properties) |
| "Multi-tenant time-sharing" | Experiment 4.1 (scalability) |
| "QoS via semantic scheduling" | Experiment 4.2 (SLA adherence) |
| "Production-ready" | Experiment 5.1 (low overhead) |
| "General across workloads" | Experiment 6.1 (6 model families) |

**Every major claim is empirically validated.**

---

## Final Recommendation

**This evaluation plan is:**
- ✅ **Achievable:** 8 weeks with 2-3 people
- ✅ **Focused:** 5 experiments, 8-10 figures, clear story
- ✅ **Rigorous:** Statistical tests, fair baselines, honest reporting
- ✅ **Compelling:** Proves core thesis with concrete evidence
- ✅ **Aligned:** Validates all claims in OVERVIEW.md

**Priority execution order:**
1. **Week 1-2:** Experiments 2.1-2.3 (LLM decode + streaming audio + conversational) - Prove semantic awareness transforms disaggregation
2. **Week 3:** Experiments 3.1-3.2 (ablation + VMU) - Component analysis + memory kernel validation
3. **Week 4:** Experiments 4.1-4.2 (scalability + QoS) - Multi-tenant validation
4. **Week 5:** Experiments 5.1, 6.1 (overhead + generality) - Framework justification + breadth

**If you execute this plan well, you will have a strong OSDI submission that clearly shows where Djinn excels.**

The key innovation: **Tailored evaluations that showcase Djinn's strengths in sequential, stateful workloads** while maintaining fair comparison through Djinn-variant baselines.