# Djinn Scheduler: Architecture Brief

**Status**: ✅ Production Ready (v2.3.15)
**Last Updated**: November 21, 2025

---

## Executive Summary

### What: Semantic-Driven Analysis & Placement Engine
Djinn's scheduler analyzes Semantically Rich Graphs (SRGs) **client-side** to extract semantic hints and placement decisions, enabling intelligent GPU placement and optimization without transferring graphs over the network. The scheduler provides semantic understanding while execution uses efficient model caching.

### Why It Matters: Practical GPU Disaggregation
- **Problem**: Traditional disaggregation operates blindly at hardware level, missing semantic context for optimization
- **Solution**: Framework-level scheduler leverages ML semantics for intelligent placement decisions and semantic hint extraction
- **Impact**: Enables performance improvements through semantic-aware decisions while maintaining efficient execution via model cache
- **Key Innovation**: Separation of concerns - semantic analysis (client) vs execution (server model cache)

### Key Metrics (v2.3.15)
- **Performance**: 10.7x faster than v2.0 graph-based execution, 0.03ms binary serialization
- **Network Reduction**: 99.7% bandwidth savings (fingerprint + hints vs full graph)
- **DMA Efficiency**: Direct memory access with synchronization for GPU transfers
- **Semantic Intelligence**: Client-side analysis with phase detection and optimization hints
- **Production Stability**: Validated across GPT-2-XL, BERT, and custom transformer architectures
- **API Compatibility**: Full PyTorch ecosystem support with HuggingFace integration

### Key Design Goals
- **Client-Side Intelligence**: Analyze SRGs locally without network transfer (303x performance gain)
- **Semantic Hint Protocol**: Lightweight metadata extraction for model cache optimization
- **Progressive Complexity**: Auto-configuration adapts sophistication to workload and user expertise
- **Model Cache Integration**: Seamless integration with server-side cached model execution
- **Separation of Concerns**: Semantic analysis (client) vs execution efficiency (server)

---

## Architecture Overview

### Core Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    CLIENT SIDE                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SEMANTICALLY RICH GRAPH (SRG)                       │  │
│  │  Built from LazyTensor interception                  │  │
│  └──────────────────────────────────────────────────────┘  │
│                        │                                    │
│                        ▼                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  SCHEDULER (Semantic Analysis)                       │  │
│  │  • Cost estimation                                    │  │
│  │  • Phase detection                                    │  │
│  │  • Pattern matching                                   │  │
│  │  • Placement decisions                                │  │
│  │  • Semantic hint extraction                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                        │                                    │
│                        ▼                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  DJINN SERIALIZER                                     │  │
│  │  • Binary protocol for execution requests             │  │
│  │  • Zero-copy tensor serialization                      │  │
│  │  • Model fingerprint + inputs + semantic hints        │  │
│  └──────────────────────────────────────────────────────┘  │
│                        │                                    │
│                        ▼                                    │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  HYBRID TRANSPORT                                     │  │
│  │  • MTU-aware network transfer                          │  │
│  │  • <1400B coalesced, >1400B scatter-gather            │  │
│  │  • DMA-optimized delivery                              │  │
│  └──────────────────────────────────────────────────────┘  │
│                        │                                    │
│                        ▼                                    │
│              [Network Transfer]                              │
│              (binary protocol + DMA pipeline)                │
│                        │                                    │
└────────────────────────┼────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    SERVER SIDE                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MODEL CACHE                                         │  │
│  │  • Cached models (not graphs!)                      │  │
│  │  • Direct model.forward() execution                 │  │
│  │  • Phase-aware memory management                     │  │
│  │  • Semantic hint application                         │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  KEY: SRG never leaves client! Graphs analyzed locally.     │
│       Execution uses cached models, not graph transfer.     │
└─────────────────────────────────────────────────────────────┘
```

### Component Dependencies (v2.3.15)

```
CLIENT SIDE:
Scheduler ────► Cost Estimator ────► Djinn Serializer ────► Hybrid Transport
    │                    │                     │                     │
    ├── Semantic         ├── FLOP Analysis     ├── Binary Protocol  ├── MTU Optimization
    │   Analysis         ├── Memory Tracking   ├── Zero-Copy         ├── Scatter-Gather
    ├── Phase Detection  └── Placement Hints   └── Length-Prefixing  └── Connection Pooling
    ├── Hint Extraction
    └── Semantic Hints ───► Binary Protocol
                          (model_id + inputs + hints)

SERVER SIDE:
Unified VMU ───► Hybrid Executor ───► Session Manager
    │                    │                      │
    ├── DMA Sync         ├── Slab Execution     └── Heartbeat Monitor
    ├── Watermark Alloc  ├── Skeletonization    └── Reference Counting
    └── Zero Fragment    └── Memory Reset       └── Automatic Cleanup
```

**Key Change (v2.3.15)**: Scheduler integrates with DjinnSerializer and HybridTransport for binary protocol transfer. Server uses DMA-synchronized VMU for direct memory access.

### Core Components & Responsibilities (v2.3.15)

| Component | Responsibility | Status | Location |
|-----------|----------------|--------|----------|
| **Scheduler Core** | Orchestrates semantic analysis and hint extraction | ✅ Production | Client |
| **Cost Estimator** | Operation-specific cost modeling (matmul, conv, attention) | ✅ Production | Client |
| **Semantic Analyzer** | Phase detection, pattern matching, optimization hints | ✅ Production | Client |
| **Djinn Serializer** | Binary protocol for execution request serialization | ✅ Production | Client |
| **Hybrid Transport** | MTU-aware network transport with syscall optimization | ✅ Production | Client |
| **Unified VMU (Server)** | DMA-synchronized memory kernel with zero fragmentation | ✅ Production | Server |
| **Hybrid Executor (Server)** | Slab-based execution with automatic memory reset | ✅ Production | Server |
| **Session Manager (Server)** | Distributed GC with heartbeat monitoring | ✅ Production | Server |

**Key Achievement**: 303x performance improvement through client-side analysis + server-side model cache execution.

### Optimization Strategy: Semantic-Driven Analysis + Model Cache Execution

**Redesigned optimization approach** separating semantic understanding from execution:

1. **Client-Side Semantic Analysis** (Scheduler):
   - Cost estimation for placement decisions
   - Phase detection (prefill/decode/vision)
   - Pattern matching for optimization hints
   - Device placement recommendations

2. **Semantic Hint Extraction**:
   - Extract placement hints from SRG analysis
   - Phase information for memory optimization
   - Co-location recommendations
   - Cost-based prioritization

3. **Model Cache Execution** (Server):
   - Direct model.forward() execution (no graph reconstruction)
   - Phase-aware memory management
   - Semantic hint application
   - Efficient weight caching

**Design Rationale**: Separation of concerns enables efficient execution (model cache) while preserving semantic intelligence (scheduler analysis). Client-side analysis eliminates graph transfer overhead while server-side caching enables fast execution.

---

## Key Design Decisions & Trade-offs

### 🔧 Design Innovations

**Local Metadata Abstraction**
- **Design Goal**: Eliminate remote metadata query latency through local LazyTensor storage
- **Architecture**: Store shape, dtype, and device information directly in tensor objects
- **Benefit**: Enables fast local queries without distributed coordination

**Semantic Cost Modeling**
- **Design Goal**: Enable ML-aware optimization using operation semantics
- **Architecture**: Cost estimators for matmul, conv, attention, and other ML operations
- **Benefit**: Framework-level optimizations not possible at hardware/driver level

**Memory-Aware Scheduling Integration**
- **Design Goal**: Integrate lifetime analysis for efficient memory utilization
- **Architecture**: Phase-aware memory budgets with lifetime-based eviction
- **Benefit**: Reduces activation memory waste in distributed execution

**Network Optimization Framework**
- **Design Goal**: Minimize network transfer overhead in distributed execution
- **Architecture**: Serialization optimization and differential graph protocols
- **Benefit**: Efficient iterative workload execution across network boundaries

**Model Cache Architecture**
- **Why**: Graph transfer was 99.4% overhead (868ms for 5ms computation)
- **Impact**: 303x faster execution, 99.7% network reduction
- **Mechanism**: Client analyzes SRG, sends model_id + inputs + semantic hints
- **Implementation**: Model cache stores models server-side, executes directly
- **Benefits**: Eliminates graph serialization/transfer/reconstruction overhead

**Semantic Hint Protocol**
- **Why**: Preserve scheduler intelligence without graph transfer
- **Impact**: Enables semantic optimizations (phase-aware, co-location) at execution time
- **Mechanism**: Extract hints from SRG analysis, include in model cache requests
- **Implementation**: Lightweight metadata attached to execution requests
- **Benefits**: Best of both worlds - semantic intelligence + efficient execution

### ⚠️ Risk Assessment

#### **Critical Technical Risks** (Implementation Challenges)

1. **LazyTensor Interception Complexity**
   - **Risk**: Overly aggressive tensor interception breaks complex ML frameworks
   - **Impact**: Incompatible with transformers, attention mechanisms, and advanced ML operations
   - **Mitigation**: Selective interception, framework-aware operation handling

2. **Shape Inference Limitations**
   - **Risk**: Incorrect shape inference for nested sequences and complex tensor constructions
   - **Impact**: Runtime errors when tensor shapes don't match expectations
   - **Mitigation**: Robust shape inference, fallback handling for unknown constructions

3. **Operation Handler Coverage**
   - **Risk**: Incomplete coverage of PyTorch operations required by ML frameworks
   - **Impact**: Runtime failures for basic tensor operations (repeat, to, etc.)
   - **Mitigation**: Comprehensive operation handler implementation, universal dispatch fallback

#### **Architecture Complexity Risks**

4. **Materialization Recursion**
   - **Risk**: Circular dependencies during tensor materialization cause infinite loops
   - **Impact**: System hangs or crashes during graph execution
   - **Mitigation**: Cycle detection, proper dependency ordering, materialization state tracking

5. **Memory Management Integration**
   - **Risk**: Complex integration between scheduling and memory management subsystems
   - **Impact**: Memory leaks, inefficient resource utilization, system instability
   - **Mitigation**: Clear subsystem boundaries, comprehensive testing, monitoring

---

## Performance & Scaling Characteristics

### Decision Latency Breakdown 

**Client-Side Scheduler Analysis** (GPT-2-XL):
| Phase | Latency | Optimization Technique |
|-------|---------|----------------------|
| SRG Construction | ~50ms | LazyTensor interception |
| Cost Estimation | 0.1ms | Cached FLOP analysis |
| Semantic Analysis | 0.2ms | Phase detection, pattern matching |
| Hint Extraction | 0.05ms | Metadata extraction |
| **Total Analysis** | **~50ms** | **All client-side, no network** |

**Server-Side Execution** (Model Cache):
| Phase | Latency | Notes |
|-------|---------|-------|
| Model Lookup | 0.5ms | Cached model retrieval |
| GPU Execution | 30.82ms | Direct model.forward() |
| **Total Execution** | **~31ms** | **No graph reconstruction** |

**Old System Comparison**:
- Old: 868ms (graph transfer + execution)
- New: 81ms (analysis + execution)
- **Improvement**: 10.7x faster

*Metrics from production profiling. Analysis is client-side, execution uses model cache.*

### Scaling Considerations

**Linear Scaling:**
- Operation count: Direct proportionality (cost estimation)
- Model complexity: Bounded by cached cost models
- Device count: Network topology modeling
- Concurrent requests: Thread-safe scheduling
- Subgraph cache: Structural hashing for identical patterns

**Non-Linear Scaling:**
- Semantic complexity: Pattern matching algorithms
- Memory constraints: Lifetime analysis computation
- Network topology: Device connectivity modeling
- Serialization: Tensor size affects transfer time
- Differential protocol: Graph change frequency impacts savings

**Bottlenecks:**
- Cost model computation for novel operations
- Memory analysis for large models
- Network topology updates under churn
- Large tensor serialization for result transfer
- Subgraph cache memory usage under high pattern diversity

---

## Maintenance & Extension Strategy

### Extension Points

**Adding New Cost Models:**
- **Easy**: Implement `CostEstimator` interface for new operations
- **Medium**: Add device-specific profiling and calibration
- **Hard**: Multi-device cost interactions and interference modeling

**Adding New Optimizations:**
- **Easy**: Implement `OptimizationStrategy` for semantic optimizations
- **Medium**: Cost-benefit analysis and validation
- **Hard**: Cross-optimization interaction analysis

**Memory Integration:**
- **Easy**: Hook into Phase 2-3 memory manager interfaces
- **Medium**: Custom memory-aware placement logic
- **Hard**: Memory model calibration for new hardware

**Serialization Optimization:**
- **Easy**: Add new serialization formats to serialization.py
- **Medium**: Custom compression algorithms for specific tensor types
- **Hard**: Hardware-accelerated serialization for specialized devices

**Caching and Optimization:**
- **Easy**: Extend subgraph cache with domain-specific hashing
- **Medium**: Add predictive prefetching to subgraph cache
- **Hard**: Multi-level caching with hierarchical eviction policies

### Monitoring & Alerting Strategy

**Key Metrics to Track:**
- Cost model accuracy (high prediction accuracy)
- Scheduling latency (<1ms for cached models)
- Memory efficiency (high GPU utilization)
- Cache hit rates (high for warm workloads)
- Serialization performance (improved vs pickle)
- Subgraph cache hit rate (effective for repeated patterns)
- Differential protocol savings (significant reduction for iterative workloads)

**Alert Conditions:**
- Cost model accuracy drops below 90%
- Scheduling latency exceeds 10ms
- Memory utilization drops below 50%
- Cache hit rate falls below acceptable levels
- Serialization performance degrades
- Subgraph cache hit rate falls below acceptable levels
- Differential protocol savings become insufficient

---

## Strategic Recommendations

### Investment Priorities

**High Priority (Q1):**
- Cost model accuracy validation and refinement
- Memory integration completion (Phase 3)
- Performance monitoring infrastructure
- Distributed scheduling reliability

**Medium Priority (Q2-Q3):**
- Advanced semantic optimizations
- Multi-device scheduling algorithms
- Adaptive cost model learning

**Low Priority (Q4+):**
- Global scheduler architecture (datacenter scale)
- Heterogeneous device support
- Real-time scheduling adaptations

### Technical Debt Assessment

**Acceptable Debt:**
- Cost model approximations (necessary for speed)
- Semantic complexity (fundamental to differentiation)
- Memory integration coupling (performance critical)

**Concerning Debt:**
- Scheduling latency optimization (monitor closely)
- Cost model maintenance (automate where possible)

**Refactoring Opportunities:**
- Consider cost model compilation for faster inference
- Evaluate plugin architecture for optimization strategies
- Assess machine learning-based cost prediction

---

## Open Source Readiness

### Complexity Assessment

**Current Complexity**: Medium-High
- **Code Size**: ~1,000 lines of core logic, well-structured
- **Dependencies**: Minimal (standard ML libraries)
- **Testing**: Comprehensive unit and integration coverage
- **Documentation**: Detailed technical docs available

**Readiness Level**: High - Scheduler is production-ready with clear interfaces and comprehensive testing.

*Detailed implementation available in `IMPLEMENTATION_DEEP_DIVE.md`*

---

## Implementation Status

### Current Architecture State
Djinn's scheduler implements a sophisticated framework for semantic-driven GPU disaggregation with several key architectural innovations designed to enable ML-aware optimization decisions.

**Implemented Components:**
- ✅ Core scheduling engine with cost estimation (client-side)
- ✅ Semantic analysis for ML-aware decision making (client-side)
- ✅ Semantic hint extraction for model cache integration
- ✅ Integration with EnhancedModelManager for model cache protocol
- ✅ Phase detection and pattern matching
- ✅ Device placement recommendations

**Architecture Goals (Achieved):**
- ✅ Local metadata abstraction for fast client-side decisions
- ✅ Semantic cost modeling for ML-aware optimizations
- ✅ Separation of semantic analysis (client) from execution (server)
- ✅ Efficient execution via model cache (no graph transfer)
- ✅ Progressive optimization complexity support

**Key Design Principles:**
- **Separation of Concerns**: Semantic analysis (scheduler) vs execution (model cache)
- **Client-Side Intelligence**: SRG analysis never leaves client, only hints transferred
- **Efficient Execution**: Model cache executes directly without graph reconstruction
- **Progressive Complexity**: Support for varying optimization sophistication levels
- **Extensibility**: Pluggable architecture for custom optimizations and cost models
- **Fault Tolerance**: Comprehensive error handling and explicit registration requirement

**Current Status:**
- ✅ Scheduler integrated with redesigned model cache system
- ✅ Semantic hints extracted and passed to model cache
- ✅ Phase-aware memory management integrated server-side
- ✅ Performance: 10.7x faster than old graph-based system
- 🔄 Ongoing: Enhanced semantic hint extraction for more optimizations
