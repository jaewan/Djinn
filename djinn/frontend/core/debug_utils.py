"""
Debug utilities for LazyTensor troubleshooting.

Used to trace operation history and understand shape transformations.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def trace_lazy_tensor_lineage(tensor: Any, depth: int = 5, prefix: str = "", visited: Optional[set] = None) -> None:
    """
    Trace LazyTensor operation history to understand shape transformations.
    
    Args:
        tensor: LazyTensor to trace
        depth: Maximum depth to trace (default: 5)
        prefix: Prefix for logging (for indentation)
        visited: Set of visited tensor IDs to avoid cycles (internal use)
    """
    if depth <= 0:
        return
    
    if visited is None:
        visited = set()
    
    # Check if this is a LazyTensor
    if not hasattr(tensor, '__class__') or type(tensor).__name__ != 'LazyTensor':
        logger.info(f"{prefix}→ Concrete tensor: {type(tensor).__name__}")
        if hasattr(tensor, 'shape'):
            logger.info(f"{prefix}  Shape: {tensor.shape}")
        return
    
    # Avoid cycles
    tensor_id = id(tensor)
    if tensor_id in visited:
        logger.info(f"{prefix}→ [CYCLE DETECTED] {getattr(tensor, 'operation', 'unknown')}")
        return
    visited.add(tensor_id)
    
    # Log this LazyTensor's operation
    operation = getattr(tensor, 'operation', 'unknown')
    shape = getattr(tensor, '_shape', None) or getattr(tensor, 'shape', 'unknown')
    logger.info(f"{prefix}→ {operation}")
    logger.info(f"{prefix}  Shape: {shape}")
    
    # Log kwargs if available (important for view operations)
    kwargs = getattr(tensor, 'kwargs', None)
    if kwargs:
        logger.info(f"{prefix}  Kwargs: {kwargs}")
    
    # Log inputs
    inputs = getattr(tensor, 'inputs', [])
    if inputs:
        logger.info(f"{prefix}  Inputs ({len(inputs)}):")
        for i, inp in enumerate(inputs):
            if hasattr(inp, '__class__') and type(inp).__name__ == 'LazyTensor':
                inp_op = getattr(inp, 'operation', 'unknown')
                inp_shape = getattr(inp, '_shape', None) or getattr(inp, 'shape', 'unknown')
                logger.info(f"{prefix}    [{i}] LazyTensor: {inp_op}, shape={inp_shape}")
            elif hasattr(inp, 'shape'):
                logger.info(f"{prefix}    [{i}] Concrete: shape={inp.shape}, type={type(inp).__name__}")
            else:
                logger.info(f"{prefix}    [{i}] {type(inp).__name__}: {inp}")
    
    # Recursively trace inputs (only LazyTensors)
    for inp in inputs:
        if hasattr(inp, '__class__') and type(inp).__name__ == 'LazyTensor':
            trace_lazy_tensor_lineage(inp, depth - 1, prefix + "  ", visited.copy())


def log_attention_mask_debug(attn_mask: Any, operation_name: str = "unknown") -> None:
    """
    Log detailed information about an attention mask for debugging.
    
    Args:
        attn_mask: Attention mask (LazyTensor or concrete tensor)
        operation_name: Name of the operation using this mask
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"ATTENTION MASK DEBUG: {operation_name}")
    logger.info(f"{'='*60}")
    logger.info(f"Mask type: {type(attn_mask).__name__}")
    
    if hasattr(attn_mask, 'shape'):
        logger.info(f"Mask shape: {attn_mask.shape}")
        logger.info(f"Mask dims: {attn_mask.dim()}")
    
    if hasattr(attn_mask, 'device'):
        logger.info(f"Mask device: {attn_mask.device}")
    
    # If it's a LazyTensor, trace its lineage
    if hasattr(attn_mask, '__class__') and type(attn_mask).__name__ == 'LazyTensor':
        logger.info(f"Is LazyTensor: True")
        logger.info(f"Operation: {getattr(attn_mask, 'operation', 'unknown')}")
        logger.info(f"Inputs: {len(getattr(attn_mask, 'inputs', []))}")
        logger.info(f"\nTracing mask lineage:")
        trace_lazy_tensor_lineage(attn_mask, depth=5)
    else:
        logger.info(f"Is LazyTensor: False")
    
    logger.info(f"{'='*60}\n")

