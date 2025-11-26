"""
LazyTensor DAG graph builder.

FX was removed because it operates at module level while Djinn needs operation-level
capture for remote execution. FX also fails on ~80% of real ML models due to dynamic
control flow. LazyTensor DAG provides the operation-level granularity required for
remote execution while working on all models.

This provides the "single source of truth" for computation graphs in Djinn.
"""

import torch
from typing import Optional, Any
import logging

from ...common.async_local import AsyncLocal
from ...core.graph_interface import Graph, LazyDAGAdapter
from .lazy_tensor import LazyTensor

logger = logging.getLogger(__name__)


# ============================================================================
# P0 OPTIMIZATION: Component Profiling Instrumentation
# ============================================================================

def get_profiler():
    """
    Get the profiler (imports here to avoid circular deps).
    
    NOTE: This is a legacy function kept for backward compatibility.
    New code should use djinn.server.profiling_context.ProfilingContext directly.
    """
    # Legacy profiling code removed - use ProfilingContext instead
    # This function returns None to avoid breaking existing code
    return None


class GraphBuilder:
    """
    Thread-local LazyTensor DAG graph builder.

    Each thread gets its own graph builder instance to avoid
    race conditions during concurrent capture.

    Strategy: Capture all tensor operations in LazyTensor DAG for remote execution.
    FX was removed because it operates at module level while Djinn needs operation-level
    capture, and FX fails on ~80% of real ML models due to dynamic control flow.

    Provides unified Graph interface through LazyDAGAdapter.

    OPTIMIZATION: Instrumented to identify bottlenecks in graph construction.
    """

    # Async-aware storage for per-context builders
    _thread_local = AsyncLocal("lazy_graph_builder")

    def __init__(self):
        # Only LazyTensor DAG - FX was removed (architectural mismatch)
        # FX operates at module level, Djinn needs operation-level capture
        # FX fails on ~80% of real ML models due to dynamic control flow
        self.root_tensor: Optional[LazyTensor] = None

    def build_from_model(self, model: torch.nn.Module, *args) -> Graph:
        """
        Build graph from model using LazyTensor DAG capture.


        Args:
            model: PyTorch model to trace
            *args: Example inputs

        Returns:
            LazyDAGAdapter with captured computation graph
        """
        profiler = get_profiler()

        # Capture using LazyTensor DAG (works on all models)
        if profiler:
            with profiler.profile_component("lazy_tensor_capture"):
                logger.info("Capturing computation graph with LazyTensor DAG...")
                output = model(*args)
        else:
            logger.info("Capturing computation graph with LazyTensor DAG...")
            output = model(*args)

        if not isinstance(output, LazyTensor):
            raise RuntimeError(
                "Model output is not a LazyTensor. "
                "Make sure tensors are created on 'remote_accelerator' device."
            )

        self.root_tensor = output
        self.use_fx = False  # FX removed - always use LazyDAG
        logger.info(f"✓ LazyTensor DAG built successfully")

        if profiler:
            with profiler.profile_component("lazy_dag_adapter"):
                return LazyDAGAdapter(self.root_tensor)
        return LazyDAGAdapter(self.root_tensor)

    def build_from_capture(self) -> Graph:
        """
        Build graph from captured LazyTensors.

        Used with context manager:
            with genie.capture():
                output = model(input)
            graph = builder.build_from_capture()

        Returns LazyDAGAdapter with the captured computation graph.
        FX was removed because it cannot reconstruct modules from operation-level
        LazyTensor DAGs, and fails on dynamic control flow common in ML models.
        """
        if self.root_tensor is None:
            raise RuntimeError("No LazyTensor captured. Use genie.capture() context first.")

        logger.info("Building graph from captured LazyTensor computation...")
        return LazyDAGAdapter(self.root_tensor)


    def add_operation(self, tensor: LazyTensor):
        """
        Register LazyTensor operation (called from LazyTensor.__init__).

        This tracks all tensors as they're created, enabling DAG construction.

        Thread-safe: Each thread has its own builder instance.
        """
        # FIX: Remove unused all_tensors tracking (memory leak)
        self.root_tensor = tensor  # Track most recent (used as output)

    def get_graph(self) -> Optional[Graph]:
        """Get the current LazyTensor DAG graph."""
        if self.root_tensor is not None:
            return LazyDAGAdapter(self.root_tensor)
        return None

    def materialize(self, target_tensor) -> torch.Tensor:
        """
        Materialize a LazyTensor by executing its computation graph.

        Design: Graph builder is responsible for graph construction and semantic
        analysis. Executor is responsible for execution. Clean separation of concerns.

        Args:
            target_tensor: LazyTensor to materialize

        Returns:
            Concrete torch.Tensor with computed values

        Note:
            This delegates to the executor module, which handles:
            - Recursive graph traversal
            - Operation execution
            - Fallback for unsupported ops
            - Error handling and reporting
        """
        from ...server.executor import execute_subgraph
        
        # Execute the graph
        result = execute_subgraph(target_tensor)
        
        # ✅ AUTOMATIC COMPACTION AFTER MATERIALIZATION
        # This prevents memory leaks in long-running workloads (e.g., LLM generation)
        try:
            from djinn.memory import get_graph_compactor, MemoryPressure, get_memory_monitor
            
            monitor = get_memory_monitor()
            stats = monitor.get_stats()
            
            # Trigger compaction if memory pressure is high or operation count threshold reached
            if stats.pressure in (MemoryPressure.HIGH, MemoryPressure.CRITICAL):
                compactor = get_graph_compactor()
                if compactor:
                    removed = compactor.compact()
                    logger.info(f"Auto-compacted graph: removed {removed} materialized nodes")
        
        except Exception as e:
            # Graceful degradation: don't fail execution if compaction has issues
            logger.debug(f"Graph compaction skipped: {e}")
        
        return result


def get_global_builder() -> GraphBuilder:
    """Get thread-local graph builder."""
    if not hasattr(GraphBuilder._thread_local, 'builder'):
        GraphBuilder._thread_local.builder = GraphBuilder()
    return GraphBuilder._thread_local.builder


def initialize_global_builder():
    """Initialize thread-local graph builder (called on import)."""
    # Create builder for main thread
    builder = get_global_builder()

    # Note: Don't set LazyTensor._graph_builder anymore since we use thread-local
    # LazyTensor instances will get the thread-local builder when needed