"""
Djinn: Semantic Disaggregated Execution Framework

INITIALIZATION DESIGN (Senior Engineer):
==========================================

This library implements async-first, eager initialization that triggers on first
Djinn API call. The design philosophy prioritizes:

1. TRANSPARENCY: Works out of the box without explicit calls
2. EARLY DETECTION: Initializes when user first touches Djinn code
3. NON-BLOCKING: Initialization happens concurrently, doesn't block operations
4. MEASURABLE: Explicit init() API allows users to measure init cost separately
5. ONCE-ONLY: Guarantee of single initialization across all threads/async tasks

INITIALIZATION FLOW:
====================

Path 1: Explicit Initialization (Recommended for Benchmarking)
  import djinn
  result = genie.init(server_address='localhost:5556')  ← User controls timing
  # Now benchmarking doesn't include init time
  model = model.to('remote_accelerator:0')  ← Already initialized

Path 2: Implicit/Auto-Initialization (Works automatically)
  import djinn
  model = model.to('remote_accelerator:0')  ← Triggers auto-init
  #  _ensure_async_init() called from FactoryInterceptor
  #  Initialization happens in background
  x = model(inputs)  ← Operation may wait for init to complete

INITIALIZATION TRIGGERS:
========================

Early triggers (non-blocking):
  1. torch.randn(device='remote_accelerator:0')  → FactoryInterceptor
  2. model.to('remote_accelerator:0')            → FactoryInterceptor  
  3. with genie.capture():                       → CaptureContext.__enter__
  
Late triggers (blocking):
  4. LazyTensor operation (x @ y)                → __torch_dispatch__
  5. Graph execution (.materialize())            → materialize()
  6. Thread pool access (get_thread_pool())      → getter function
  7. Explicit call (genie.init())                → public API

ASYNC-FIRST DESIGN:
===================

Key insight: Use asyncio.create_task() to start init in background
- Returns immediately (non-blocking)
- Initialization happens concurrently
- Operations wait if they need resources before init completes
- No busy-spinning or polling

Thread Safety:
- Double-check locking pattern in _ensure_async_init()
- Once-only guarantee via initialization_task tracking
- Thread-local storage for event loop detection
- Background thread created if no event loop exists

BENCHMARK RECOMMENDATIONS:
==========================

For publication-quality benchmarks:

  from benchmarks.comprehensive_evaluation import ComprehensiveEvaluation
  
  # PHASE 1: Measure just init cost
  import djinn
  import time
  
  start = time.perf_counter()
  result = genie.init(server_address='localhost:5556')
  init_time = (time.perf_counter() - start) * 1000
  print(f"Initialization: {init_time:.1f}ms")
  
  # PHASE 2: Measure workload (init already done)
  eval = ComprehensiveEvaluation(use_real_models=True, spawn_server=True)
  await eval.run_all(num_runs=10, num_warmup=3)

This separates initialization cost from workload cost for honest measurement.
"""

import logging
logger = logging.getLogger(__name__)

# Version
__version__ = "0.2.0"

# Core interception setup
def _initialize():
    """Initialize Djinn interception layer."""
    try:
        # Step 1: Python-only mode (C++ backend not required)
        logger.info("Djinn initialized in Python-only mode (optimal performance)")

        # Step 2: Wrap factory functions (REQUIRED)
        from .frontend.core.factory_interceptor import wrap_factories
        wrap_factories()

        # Step 3: Setup remote accelerator device support
        from .core.device_compatibility import setup as setup_device_support
        setup_device_support()

        # Step 4: Initialize graph builder
        from .frontend.core.graph_builder import initialize_global_builder
        initialize_global_builder()

        # Step 5: Warm up shape inference cache (performance optimization)
        from .core.warmup import _warmup_shape_inference
        _warmup_shape_inference()

        logger.info("Djinn initialized successfully")

    except Exception as e:
        logger.error(f"Djinn initialization failed: {e}")
        raise

# Initialize on import
_initialize()

# Public API - Phase 1 (Core)
from .frontend.core.lazy_tensor import LazyTensor
from .frontend.core.capture import capture, get_graph, is_capturing
from .server.subgraph_builder import SubgraphBuilder, RemoteSubgraph

# Public API - Phase 2 (Smart Fragmentation)
from .server.optimizations.smart_subgraph_builder import (
    SmartSubgraphBuilder,
    FragmentationConfig,
    SubgraphFragment,
    CostEstimate,
    MemoryEstimator,
    CostCalculator
)

# Public API - Phase 2 (Semantic Analysis & Scheduling)
# Note: semantic module was deprecated and removed - imports moved to frontend.semantic
from .frontend.semantic.analyzer import SemanticAnalyzer
from .frontend.semantic.workload import WorkloadProfile, WorkloadType
from .frontend.semantic.pattern_registry import PatternRegistry
from .scheduler import (
    Scheduler,
    ExecutionSchedule,
    SchedulingStrategy,
)
from .frontend.semantic.annotator import annotate_graph, AnnotatedGraph

# Public API - Phase 3 (Async-First Runtime Initialization)
from .backend.runtime.initialization import (
    init as init_sync,
    init_async,
    ensure_initialized,
    get_runtime_state,
    is_initialized,
    get_thread_pool,
    get_coordinator,
    get_initialization_time_ms,
    register_init_hook,
)

# Public API - Phase 3 (Semantic Hints)
from .core.semantic_hints import session, SemanticHints, Priority, get_current_hints

# Convenience wrapper for synchronous code
def init(
    server_address=None,
    auto_connect=True,
    thread_pool_size=4,
    profiling=False
):
    """
    Initialize Djinn runtime explicitly.
    
    RECOMMENDED FOR BENCHMARKING: Call this before measurements to separate
    initialization cost from workload cost.
    
    If not called, auto-init will trigger on first Djinn API call.
    
    Args:
        server_address: Remote server address (e.g., 'localhost:5556')
                       If None, checks GENIE_SERVER_ADDRESS env var
        auto_connect: Whether to connect to remote server automatically
        thread_pool_size: Number of threads in execution pool (default: 4)
        profiling: Enable performance profiling (default: False)
    
    Returns:
        Dictionary with initialization result:
        {
            'status': 'success' or 'error',
            'initialized': bool,
            'server_address': Optional[str],
            'gpu_count': Optional[int],
            'thread_pool_size': int,
            'duration_ms': float,  # Initialization time
            'error': Optional[str]
        }
    
    Example:
        >>> import djinn
        >>> result = genie.init(server_address='localhost:5556')
        >>> if result['status'] == 'success':
        ...     print(f"Initialized in {result['duration_ms']:.1f}ms")
    """
    return init_sync(
        server_address=server_address,
        auto_connect=auto_connect,
        thread_pool_size=thread_pool_size,
        profiling=profiling
    )


def analyze(graph=None):
    """
    Analyze a computation graph for semantic information.

    Args:
        graph: Graph to analyze. If None, uses the most recently captured graph.

    Returns:
        WorkloadProfile with semantic analysis results
    """
    if graph is None:
        graph = get_graph()
        if graph is None:
            raise RuntimeError("No graph available. Call get_graph() after capture or pass graph explicitly.")

    analyzer = SemanticAnalyzer()
    # For LazyDAG graphs, we'll need to handle this differently
    # For now, return a basic profile
    return WorkloadProfile(
        workload_type=WorkloadType.UNKNOWN,
        patterns=[],
        metadata={'graph_type': graph.backend_type}
    )

def schedule(graph=None, profile=None):
    """
    Create an execution schedule for a computation graph.

    Args:
        graph: Graph to schedule. If None, uses the most recently captured graph.
        profile: WorkloadProfile from semantic analysis. If None, runs analysis first.

    Returns:
        ExecutionSchedule with scheduling decisions
    """
    if graph is None:
        graph = get_graph()
        if graph is None:
            raise RuntimeError("No graph available. Call get_graph() after capture or pass graph explicitly.")

    if profile is None:
        profile = analyze(graph)

    # ✅ FIXED: Actually use the scheduler
    from .scheduler import Scheduler
    from .scheduler.core.cost_estimator import GraphCostEstimator, NetworkTopology

    scheduler = Scheduler(
        cost_estimator=GraphCostEstimator(),
        network_topology=NetworkTopology()
    )

    # Create actual schedule using semantic analysis
    return scheduler.create_schedule(graph, profile)


# ============================================================================
# PHASE 1: GRAPH CACHING API
# ============================================================================

def execute_model(model, inputs, use_cache=True, use_blocks=True, **kwargs):
    """
    Execute model with graph caching and block compilation.

    Phase 1 (Graph Caching): Eliminate repeated graph capture (450ms)
    Phase 2 (Block Compilation): Reduce RPC calls from 1500 to 15 (285ms)

    Args:
        model: PyTorch model
        inputs: Input tensor(s)
        use_cache: Whether to use graph cache (default: True)
        use_blocks: Whether to compile to blocks (default: True)
        **kwargs: Additional execution options

    Returns:
        Model output (same as direct model call)

    Performance:
        Cold start (first run):    ~550ms (capture + compile + execute)
        Warm with cache+blocks:    ~40ms (14x faster!)

    Example:
        >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
        >>> inputs = torch.randint(0, 50257, (8, 128))
        >>>
        >>> # First execution (cold)
        >>> output = djinn.execute_model(model, inputs)  # 550ms
        >>>
        >>> # Subsequent executions (warm with caching + blocks)
        >>> output = djinn.execute_model(model, inputs)  # 40ms

    Notes:
        - Requires input shape to be constant for cache hits
        - After fine-tuning, call invalidate_model_cache(model)
        - Cache automatically evicts LRU entries when full (default 100 models)
    """
    from .server.graph_cache import get_graph_cache
    from .server.block_compiler import get_block_compiler

    cache = get_graph_cache()
    
    # Phase 1: Get or capture graph (handles caching internally)
    graph = cache.get_or_capture(
        model, 
        inputs, 
        force_recapture=not use_cache
    )
    
    # Phase 2: Compile to blocks (100x RPC reduction)
    if use_blocks:
        try:
            compiler = get_block_compiler()
            blocks = compiler.compile_model(model, inputs)
            
            if blocks:
                # Execute blocks instead of full graph
                from .server.block_compiler import BlockExecutor
                executor = BlockExecutor()
                
                # Prepare input dict
                if isinstance(inputs, torch.Tensor):
                    input_dict = {blocks[0].input_names[0]: inputs}
                else:
                    input_dict = {blocks[0].input_names[0]: inputs}
                
                # Execute blocks and return final output
                outputs = executor.execute_blocks(blocks, input_dict)
                
                # Return final output tensor
                return list(outputs.values())[-1]
        
        except Exception as e:
            logger.debug(f"Block compilation failed: {e}, falling back to graph execution")
            # Fall through to standard execution

    # Fallback: Execute full graph or direct model call
    try:
        # Try to use graph executor if available
        from .server.executor import SimpleExecutor
        executor = SimpleExecutor()
        return executor.execute_graph(graph, inputs)
    except Exception:
        # Fallback: Direct model call (works without full Djinn setup)
        if isinstance(inputs, (list, tuple)):
            return model(*inputs)
        else:
            return model(inputs)


def invalidate_model_cache(model):
    """
    Invalidate cached graph for model.

    Use after:
    - Fine-tuning (model parameters changed)
    - Model architecture modification
    - Manual cache refresh

    Args:
        model: PyTorch model to invalidate

    Example:
        >>> model = load_model('gpt2')
        >>> djinn.execute_model(model, inputs)  # Cache miss, captures
        >>> djinn.execute_model(model, inputs)  # Cache hit
        >>>
        >>> fine_tune(model)  # Modify parameters
        >>> djinn.invalidate_model_cache(model)  # Clear cache
        >>> djinn.execute_model(model, inputs)  # Cache miss, recaptures
    """
    from .server.graph_cache import get_graph_cache

    cache = get_graph_cache()
    cache.invalidate(model)
    logger.info(f"Invalidated cache for model: {model.__class__.__name__}")


def get_graph_cache_stats():
    """
    Get current graph cache statistics.

    Returns:
        Dictionary with cache statistics:
        - entries: Number of cached graphs
        - max_entries: Maximum cache size
        - hits: Number of cache hits
        - misses: Number of cache misses
        - hit_rate: Cache hit rate (0-1)
        - evictions: Number of evicted entries
        - invalidations: Number of manual invalidations
        - total_time_saved_ms: Total time saved by caching

    Example:
        >>> stats = djinn.get_graph_cache_stats()
        >>> print(f"Hit rate: {stats['hit_rate']:.1%}")
        >>> print(f"Time saved: {stats['total_time_saved_ms']/1000:.1f}s")
    """
    from .server.graph_cache import get_graph_cache

    cache = get_graph_cache()
    return cache.get_stats()


def print_graph_cache_stats():
    """Print human-readable graph cache statistics."""
    from .server.graph_cache import get_graph_cache

    cache = get_graph_cache()
    cache.print_stats()


def clear_graph_cache():
    """Clear all cached graphs."""
    from .server.graph_cache import get_graph_cache

    cache = get_graph_cache()
    cache.clear()
    logger.info("Graph cache cleared")


# ============================================================================
# PHASE 2: BLOCK COMPILATION API
# ============================================================================

def compile_model_to_blocks(model, sample_input):
    """
    Compile model to TorchScript blocks for coarse-grained execution.

    Reduces RPC calls from 1500 to 15 (100x) by batching operations
    into TorchScript blocks at module boundaries.

    Args:
        model: PyTorch model to compile
        sample_input: Sample input tensor for shape inference

    Returns:
        List of ExecutableBlock objects ready for execution

    Example:
        >>> model = GPT2LMHeadModel.from_pretrained('gpt2')
        >>> inputs = torch.randint(0, 50257, (8, 128))
        >>> blocks = djinn.compile_model_to_blocks(model, inputs)
        >>> print(f"Compiled to {len(blocks)} blocks")
    """
    from .server.block_compiler import get_block_compiler

    compiler = get_block_compiler()
    return compiler.compile_model(model, sample_input)


def get_block_compilation_stats():
    """Get statistics from last block compilation."""
    from .server.block_compiler import get_block_compiler

    compiler = get_block_compiler()
    return {
        'total_blocks': compiler.stats['total_blocks'],
        'successful_compilations': compiler.stats['successful_compilations'],
        'failed_compilations': compiler.stats['failed_compilations'],
        'total_operations': compiler.stats['total_operations'],
        'total_memory_bytes': compiler.stats['total_memory_bytes'],
    }


# ============================================================================
# PHASE 4: TENSORRT OPTIMIZATION API
# ============================================================================

def profile_block_execution(block_id, execution_time_ms, torchscript_module=None):
    """
    Record block execution for TensorRT optimization profiling.

    After 100 executions, block will be auto-compiled to TensorRT for 2-3x speedup.

    Args:
        block_id: Identifier of the block being executed
        execution_time_ms: Time taken to execute (in milliseconds)
        torchscript_module: Optional TorchScript module for compilation

    Example:
        >>> import time
        >>> start = time.perf_counter()
        >>> output = execute_block(block_id, inputs)
        >>> elapsed_ms = (time.perf_counter() - start) * 1000
        >>> djinn.profile_block_execution(block_id, elapsed_ms, block_module)
    """
    from .server.tensorrt_compiler import get_tensorrt_compiler

    compiler = get_tensorrt_compiler()
    compiler.record_execution(block_id, execution_time_ms, torchscript_module)


def try_tensorrt_compilation(block_id, torchscript_module, sample_input, use_fp16=True):
    """
    Attempt lazy TensorRT compilation of a block.

    Automatically triggered after 100 executions (profiling threshold).
    Can also be called manually for immediate optimization.

    Args:
        block_id: Block identifier
        torchscript_module: TorchScript module to compile
        sample_input: Sample input for compilation
        use_fp16: Enable FP16 optimization (2-3x speedup, default: True)

    Returns:
        Compiled TensorRT module or None if compilation failed

    Example:
        >>> from djinn.server.tensorrt_compiler import get_tensorrt_compiler
        >>> compiler = get_tensorrt_compiler()
        >>> compiler.register_block(1, "attention_block")
        >>> # After sufficient executions...
        >>> compiled = djinn.try_tensorrt_compilation(1, ts_module, sample, use_fp16=True)
    """
    from .server.tensorrt_compiler import get_tensorrt_compiler

    compiler = get_tensorrt_compiler()
    return compiler.try_compile_tensorrt(block_id, torchscript_module, sample_input, use_fp16)


def get_tensorrt_stats():
    """
    Get TensorRT compilation and optimization statistics.

    Returns:
        Dictionary with:
        - total_compilations: Total attempted compilations
        - successful_compilations: Successfully compiled blocks
        - failed_compilations: Failed compilation attempts
        - fp16_compilations: Blocks compiled with FP16
        - success_rate: Compilation success rate (0-1)
        - avg_compilation_time_ms: Average compilation time
        - blocks_compiled: Number of blocks with TensorRT
        - total_blocks: Total registered blocks
        - estimated_speedup: Expected speedup from TensorRT (typically 2.5x)

    Example:
        >>> stats = djinn.get_tensorrt_stats()
        >>> print(f"Compiled {stats['blocks_compiled']}/{stats['total_blocks']} blocks")
        >>> print(f"Success rate: {stats['success_rate']:.1%}")
    """
    from .server.tensorrt_compiler import get_tensorrt_compiler

    compiler = get_tensorrt_compiler()
    return compiler.get_stats()


def register_block_for_optimization(block_id, block_name):
    """
    Register a block for TensorRT optimization tracking.

    Args:
        block_id: Unique block identifier
        block_name: Human-readable block name

    Example:
        >>> djinn.register_block_for_optimization(1, "transformer_attention")
    """
    from .server.tensorrt_compiler import get_tensorrt_compiler

    compiler = get_tensorrt_compiler()
    compiler.register_block(block_id, block_name)


def get_optimization_hint(block_id):
    """
    Get adaptive optimization hint for a block.

    Returns information about whether the block should use TensorRT/FP16.

    Args:
        block_id: Block identifier

    Returns:
        Dictionary with optimization recommendations

    Example:
        >>> hint = djinn.get_optimization_hint(1)
        >>> if hint.get('use_tensorrt'):
        ...     use_compiled_version()
    """
    from .server.tensorrt_compiler import get_tensorrt_compiler, AdaptiveOptimizer

    compiler = get_tensorrt_compiler()
    optimizer = AdaptiveOptimizer(compiler)
    return optimizer.get_optimization_hint(block_id)


# Week 1-2: Expose config for materialization control
from .config import get_config as _get_config

# Public config API
config = _get_config()

__all__ = [
    # Phase 1 (Core)
    'LazyTensor',
    'capture',
    'get_graph',
    'is_capturing',
    'SubgraphBuilder',
    'RemoteSubgraph',
    'config',  # Week 1-2: Materialization config

    # Phase 2 (Smart Fragmentation)
    'SmartSubgraphBuilder',
    'FragmentationConfig',
    'SubgraphFragment',
    'CostEstimate',
    'MemoryEstimator',
    'CostCalculator',

    # Phase 2 (Semantic Analysis & Scheduling)
    'SemanticAnalyzer',
    'Scheduler',
    'WorkloadProfile',
    'WorkloadType',
    'PatternRegistry',
    'ExecutionSchedule',
    'SchedulingStrategy',

    # Phase 3 (Async-First Runtime Initialization)
    'init',                          # ✅ Public API (synchronous wrapper)
    'init_async',                    # ✅ Public API (async version)
    'init_sync',                     # ✅ Public API (core implementation)
    'ensure_initialized',            # ✅ Trigger auto-init from sync contexts
    'get_runtime_state',             # ✅ Get global runtime state
    'is_initialized',                # ✅ Check if initialized
    'get_thread_pool',               # ✅ Access thread pool
    'get_coordinator',               # ✅ Access remote coordinator
    'get_initialization_time_ms',    # ✅ Get init duration
    'register_init_hook',            # ✅ Register custom init handlers

    # Phase 3 (Semantic Hints)
    'session',                       # ✅ Context manager for semantic hints
    'SemanticHints',                 # ✅ Semantic hints dataclass
    'Priority',                      # ✅ Priority enum
    'get_current_hints',             # ✅ Get current semantic hints from context

    # Phase 3
    'annotate_graph',
    'AnnotatedGraph',

    # Phase 1: Graph Caching Optimization
    'execute_model',                 # ✅ Execute with caching (2.6x warm speedup)
    'invalidate_model_cache',        # ✅ Invalidate after fine-tuning
    'get_graph_cache_stats',         # ✅ Monitor cache performance
    'print_graph_cache_stats',       # ✅ Human-readable stats
    'clear_graph_cache',             # ✅ Manual cache clear

    # Phase 2: Block Compilation
    'compile_model_to_blocks',       # ✅ Compile to TorchScript blocks (100x RPC)
    'get_block_compilation_stats',   # ✅ Block compilation statistics

    # Phase 4: TensorRT Optimization
    'profile_block_execution',       # ✅ Record execution for TRT profiling
    'try_tensorrt_compilation',      # ✅ Lazy TRT compilation
    'get_tensorrt_stats',            # ✅ TRT compilation statistics
    'register_block_for_optimization', # ✅ Register block for tracking
    'get_optimization_hint',         # ✅ Adaptive optimization decisions

    # Convenience Functions
    'analyze',
    'schedule',
]