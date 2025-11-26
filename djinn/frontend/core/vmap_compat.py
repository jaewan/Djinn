"""
Automatic vmap compatibility for LazyTensors.

This module provides transparent vmap support by automatically materializing
LazyTensors before they enter vmap's batching mechanism. Completely transparent
to users - no code changes or configuration needed.

Installed automatically during Djinn initialization.
"""

import torch
from functools import wraps
from typing import Callable, Any, Union, Optional

_original_vmap = None
_installed = False


def _auto_materialize_vmap(
    func: Callable,
    in_dims: Union[int, tuple] = 0,
    out_dims: Union[int, tuple] = 0,
    randomness: str = 'error',
    chunk_size: Optional[int] = None
) -> Callable:
    """
    Wrapper that automatically materializes LazyTensors before vmap execution.
    
    Completely transparent - user sees no difference. Materialization happens
    automatically when LazyTensors are detected in arguments.
    """
    # Import here to avoid circular dependency
    from .lazy_tensor import LazyTensor
    
    @wraps(func)
    def materialized_func(*args, **kwargs):
        """Materialize any LazyTensor arguments before calling func."""
        # Materialize LazyTensors in args
        materialized_args = tuple(
            arg.materialize() if isinstance(arg, LazyTensor) else arg
            for arg in args
        )
        
        # Materialize LazyTensors in kwargs
        materialized_kwargs = {
            k: v.materialize() if isinstance(v, LazyTensor) else v
            for k, v in kwargs.items()
        }
        
        return func(*materialized_args, **materialized_kwargs)
    
    # Call original vmap with our wrapper
    return _original_vmap(
        materialized_func,
        in_dims=in_dims,
        out_dims=out_dims,
        randomness=randomness,
        chunk_size=chunk_size
    )


def install():
    """
    Install vmap wrapper (called during Djinn initialization).
    
    This replaces torch.vmap with our wrapper that handles LazyTensors.
    Completely transparent to users - no API changes.
    """
    global _original_vmap, _installed
    
    if _installed:
        return
    
    _original_vmap = torch.vmap
    torch.vmap = _auto_materialize_vmap
    _installed = True


# Convenience function for initialization
def install_vmap_compatibility():
    """Install vmap compatibility (alias for install())."""
    install()
