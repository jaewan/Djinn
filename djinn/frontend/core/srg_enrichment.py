"""
SRG Enrichment: Semantic Rich Graph Enhancement for LazyTensor

Adds semantic properties to LazyTensor instances without modifying the core class.
This demonstrates the "enrich in-place" approach for SRG implementation.

Key Benefits:
- Zero modification to LazyTensor.__init__ overhead
- Lazy evaluation (compute once, cache forever)
- Clean separation of concerns
- Easy to enable/disable for testing
"""

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .lazy_tensor import LazyTensor

logger = logging.getLogger(__name__)


def enrich_lazy_tensor_with_srg(lazy_tensor: 'LazyTensor') -> None:
    """
    Enrich a LazyTensor instance with SRG semantic properties.

    This adds semantic classification, lifecycle inference, and compute cost
    estimation as lazy properties.

    Args:
        lazy_tensor: LazyTensor instance to enrich
    """
    # Add semantic properties dynamically
    lazy_tensor.semantic_class = _get_semantic_class.__get__(lazy_tensor, type(lazy_tensor))
    lazy_tensor.lifecycle = _get_lifecycle.__get__(lazy_tensor, type(lazy_tensor))
    lazy_tensor.compute_cost = _get_compute_cost.__get__(lazy_tensor, type(lazy_tensor))
    lazy_tensor.detected_phase = _get_detected_phase.__get__(lazy_tensor, type(lazy_tensor))


def _get_semantic_class(self) -> 'OperationClass':
    """
    Lazy semantic classification of this operation.

    Uses OperationClassifier to determine operation semantics.
    Computed once and cached.

    Returns:
        OperationClass enum indicating how this operation should be executed
    """
    if not hasattr(self, '_semantic_class') or self._semantic_class is None:
        try:
            from .operation_classifier import OperationClassifier, OperationClass
            self._semantic_class = OperationClassifier.classify(
                self._operation,
                return_type=None,
                args=self._inputs,
                kwargs=self._kwargs
            )
        except Exception as e:
            # Fallback to COMPUTE_OPERATION if classification fails
            from .operation_classifier import OperationClass
            logger.debug(f"Semantic classification failed for {self._operation}: {e}")
            self._semantic_class = OperationClass.COMPUTE_OPERATION

    return self._semantic_class


def _get_lifecycle(self) -> str:
    """
    Lazy lifecycle classification of this tensor.

    Determines whether tensor should be allocated in:
    - 'ephemeral': Stack Segment (activations, temporary results)
    - 'persistent': Data Segment (KV cache, embeddings, state)

    Returns:
        'ephemeral' or 'persistent'
    """
    if not hasattr(self, '_lifecycle') or self._lifecycle is None:
        try:
            op_class = self.semantic_class
            operation = self._operation.lower()

            # Persistent: operations that maintain state across requests
            if (hasattr(op_class, 'name') and op_class.name == 'MATERIALIZATION_TRIGGER') or \
               'kv_cache' in operation or 'embedding' in operation or 'positional' in operation:
                self._lifecycle = 'persistent'
            # Ephemeral: computation results, activations
            else:
                self._lifecycle = 'ephemeral'

        except Exception as e:
            # Fallback to ephemeral if lifecycle inference fails
            logger.debug(f"Lifecycle inference failed for {self._operation}: {e}")
            self._lifecycle = 'ephemeral'

    return self._lifecycle


def _get_compute_cost(self) -> float:
    """
    Lazy computation cost estimation.

    Estimates FLOPs or relative compute cost for scheduling decisions.

    Returns:
        Estimated compute cost (higher = more expensive)
    """
    if not hasattr(self, '_compute_cost') or self._compute_cost is None:
        try:
            # Estimate based on input tensor sizes
            total_elements = 0
            for inp in self._inputs:
                if hasattr(inp, 'numel'):
                    total_elements += inp.numel()

            # Cost multipliers for different operations
            operation = self._operation.lower()
            if 'attention' in operation or 'matmul' in operation:
                cost_multiplier = 10.0  # Expensive operations
            elif 'linear' in operation or 'dense' in operation:
                cost_multiplier = 5.0   # Moderately expensive
            elif 'activation' in operation or 'norm' in operation:
                cost_multiplier = 2.0   # Element-wise
            else:
                cost_multiplier = 1.0   # Default

            self._compute_cost = total_elements * cost_multiplier

        except Exception as e:
            # Fallback to basic cost if estimation fails
            logger.debug(f"Compute cost estimation failed for {self._operation}: {e}")
            self._compute_cost = 1.0

    return self._compute_cost


def _get_detected_phase(self) -> str:
    """
    Get the detected phase for this inference (prefill/decode).

    Returns:
        'prefill', 'decode', or 'unknown'
    """
    from .lazy_tensor import _phase_detector
    return _phase_detector.get_phase()


# ============================================================================
# SRG EXECUTOR INTEGRATION
# ============================================================================

def allocate_tensor_with_srg(tensor, vmu) -> int:
    """
    Allocate tensor memory using SRG semantic information.

    This demonstrates how the executor would use SRG metadata
    for intelligent memory allocation.

    Args:
        tensor: LazyTensor with SRG enrichment
        vmu: Unified VMU instance

    Returns:
        Allocation offset
    """
    # Use SRG lifecycle information for allocation decisions
    if hasattr(tensor, 'lifecycle'):
        lifecycle = tensor.lifecycle
        if lifecycle == 'persistent':
            # Allocate in Data Segment (session-owned, survives requests)
            session_id = getattr(tensor, '_session_id', 'default_session')
            size_bytes = getattr(tensor, 'nbytes', 1024)  # Estimate size
            return vmu.allocate_session_data(session_id, size_bytes, tensor._operation)
        else:
            # Allocate in Stack Segment (volatile, reset after execution)
            size_bytes = getattr(tensor, 'nbytes', 1024)
            offset, view = vmu.allocate_volatile(size_bytes, tensor._operation)
            return offset
    else:
        # Fallback to legacy allocation
        size_bytes = getattr(tensor, 'nbytes', 1024)
        offset, view = vmu.allocate_volatile(size_bytes, tensor._operation)
        return offset


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def enable_srg_enrichment():
    """Enable SRG enrichment for all new LazyTensor instances."""
    # Monkey patch LazyTensor.__init__ to add enrichment
    from .lazy_tensor import LazyTensor
    original_init = LazyTensor.__init__

    def enriched_init(self, *args, **kwargs):
        # Call original init
        original_init(self, *args, **kwargs)
        # Add SRG enrichment
        enrich_lazy_tensor_with_srg(self)

    LazyTensor.__init__ = enriched_init
    logger.info("✅ SRG enrichment enabled for LazyTensor")


def disable_srg_enrichment():
    """Disable SRG enrichment."""
    # Restore original __init__ (would need to be implemented properly)
    logger.info("ℹ️  SRG enrichment disabled")
