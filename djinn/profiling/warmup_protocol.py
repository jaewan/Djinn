"""
Warmup protocol for consistent profiling measurements.

Warms up JIT compilers, caches, and GPU kernels before profiling.
"""

import logging
import time
from typing import Callable, Any
import torch

logger = logging.getLogger(__name__)


def warmup(func: Callable, *args, num_runs: int = 3, **kwargs) -> Any:
    """
    Warmup a function by running it multiple times before actual profiling.
    
    Args:
        func: Function to warmup
        *args: Positional arguments for func
        num_runs: Number of warmup runs (default: 3)
        **kwargs: Keyword arguments for func
    
    Returns:
        Result from the last warmup run
    """
    logger.debug(f"Warming up {func.__name__} with {num_runs} runs...")
    
    result = None
    for i in range(num_runs):
        try:
            result = func(*args, **kwargs)
            # Sync GPU if available
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            logger.warning(f"Warmup run {i+1} failed: {e}")
            if i == num_runs - 1:
                raise
    
    logger.debug(f"Warmup complete for {func.__name__}")
    return result


class WarmupContext:
    """Context manager for warmup protocol."""
    
    def __init__(self, num_runs: int = 3):
        self.num_runs = num_runs
        self.warmed_up = False
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def run(self, func: Callable, *args, **kwargs) -> Any:
        """Run function with warmup."""
        if not self.warmed_up:
            warmup(func, *args, num_runs=self.num_runs, **kwargs)
            self.warmed_up = True
        
        return func(*args, **kwargs)
