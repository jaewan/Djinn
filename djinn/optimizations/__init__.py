"""
Unified Optimizations API for Djinn.

Provides consistent interfaces to all optimization functionality:
- Operation batching (small operation grouping)
- Fusion analysis (optimization opportunity detection)

Usage:
    from djinn.optimizations import get_optimizer

    # Get operation batcher
    batcher = get_optimizer('batching')

    # Get fusion analyzer
    analyzer = get_optimizer('fusion')
"""

from typing import Union, Any
from .operation_batching import OperationBatcher, get_operation_batcher
from .fusion_analyzer import OperationFusionAnalyzer, get_operation_fusion_analyzer

# Re-export for convenience
__all__ = [
    'get_optimizer',
    'OperationBatcher',
    'OperationFusionAnalyzer'
]


def get_optimizer(optimizer_type: str, **kwargs) -> Union[OperationBatcher, OperationFusionAnalyzer]:
    """
    Unified optimizer factory function.

    Args:
        optimizer_type: Type of optimizer ('batching', 'fusion')
        **kwargs: Optimizer-specific initialization parameters

    Returns:
        Optimizer instance

    Raises:
        ValueError: If optimizer_type is not supported
    """
    if optimizer_type in ('batching', 'batch'):
        return get_operation_batcher()
    elif optimizer_type in ('fusion', 'analyzer'):
        return get_operation_fusion_analyzer()
    else:
        raise ValueError(
            f"Unsupported optimizer type: {optimizer_type}. "
            f"Supported types: 'batching', 'fusion'"
        )

