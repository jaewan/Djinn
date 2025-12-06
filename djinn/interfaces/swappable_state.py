"""
SwappableState: Abstract interface for any state that can be swapped GPU<->CPU.

This enables Djinn to work with any tensor state representation:
- HuggingFace DynamicCache
- vLLM PagedAttention
- TensorRT-LLM KV blocks
- Custom activation checkpoints
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict


class SwappableState(ABC):
    """
    Interface for any state that can be swapped between GPU and CPU memory.
    
    This is the core abstraction that enables Djinn's memory virtualization:
    different frameworks represent KV caches differently, but all must be
    convertible to/from a CPU-transferable format.
    """
    
    @abstractmethod
    def to_host_format(self) -> Tuple[Any, Dict[str, Any]]:
        """
        Convert state to CPU-transferable format.
        
        Returns:
            Tuple of:
            - cpu_data: Actual data to store on CPU (tensor, tuple, dict, etc.)
            - metadata: Dictionary with reconstruction hints
            
        Example:
            kv_on_gpu = DynamicCache(...)  # 2GB on GPU
            cpu_data, meta = kv_on_gpu.to_host_format()
            # cpu_data: list of pinned CPU tensors
            # meta: {'type': 'dynamic_cache', 'num_layers': 32}
        """
        pass
    
    @classmethod
    @abstractmethod
    def from_host_format(cls, cpu_data: Any, metadata: Dict[str, Any]) -> 'SwappableState':
        """
        Reconstruct state from CPU format back to active representation.
        
        Args:
            cpu_data: Data returned by to_host_format()
            metadata: Metadata returned by to_host_format()
        
        Returns:
            Reconstructed state, ready for GPU execution
            
        Example:
            kv_cpu = DynamicCache.from_host_format(cpu_data, meta)
            # Now on CPU but reconstructed to proper type
        """
        pass
    
    @abstractmethod
    def gpu_size_bytes(self) -> int:
        """
        Report GPU memory footprint in bytes.
        
        Used for memory budgeting and eviction decisions.
        Must reflect actual GPU VRAM used, not logical size.
        """
        pass

