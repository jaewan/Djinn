"""
Type checking utilities for performance-critical paths.

Provides optimized type checking functions that avoid the overhead
of isinstance() in hot execution paths.
"""

import torch
from typing import Any

__all__ = ['is_tensor', 'is_tensor_fast', 'TensorTypeCache']


# Cache torch.Tensor type for faster checks
_TENSOR_TYPE = type(torch.tensor(0))
_TENSOR_CLASS = torch.Tensor


class TensorTypeCache:
    """
    Cache for tensor type checks to avoid repeated isinstance() calls.
    
    Usage:
        cache = TensorTypeCache()
        if cache.is_tensor(obj):
            # Fast path
    """
    
    def __init__(self):
        self._tensor_type = type(torch.tensor(0))
        self._tensor_class = torch.Tensor
    
    def is_tensor(self, obj: Any) -> bool:
        """
        Fast tensor type check using cached type.
        
        This is faster than isinstance() for repeated checks in hot paths.
        """
        # Use type() comparison for speed (faster than isinstance)
        obj_type = type(obj)
        return obj_type is self._tensor_type or isinstance(obj, self._tensor_class)
    
    def is_tensor_strict(self, obj: Any) -> bool:
        """
        Strict tensor check (only exact type, no subclasses).
        
        Fastest check, but doesn't handle tensor subclasses.
        """
        return type(obj) is self._tensor_type


def is_tensor(obj: Any) -> bool:
    """
    Fast tensor type check using cached type.
    
    This is faster than isinstance() for repeated checks in hot paths.
    Optimized for performance-critical code.
    
    Args:
        obj: Object to check
        
    Returns:
        True if obj is a torch.Tensor
    """
    # Use type() comparison first (faster for exact matches)
    obj_type = type(obj)
    if obj_type is _TENSOR_TYPE:
        return True
    # Fallback to isinstance for subclasses
    return isinstance(obj, _TENSOR_CLASS)


def is_tensor_fast(obj: Any) -> bool:
    """
    Fastest tensor check (exact type only, no subclasses).
    
    Use this when you know you're dealing with standard torch.Tensor
    instances, not custom subclasses.
    
    Args:
        obj: Object to check
        
    Returns:
        True if obj is exactly torch.Tensor (not a subclass)
    """
    return type(obj) is _TENSOR_TYPE

