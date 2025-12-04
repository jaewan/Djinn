"""
Host Swap Pool v2: Simplified pinned host memory management using PyTorch's allocator.

Purpose:
- Provide pinned memory for KV cache eviction (swap-to-CPU)
- Track which sessions are swapped
- Enable fast restore with DMA transfers

Architecture:
- SIMPLIFIED: Delegate allocation to PyTorch's CachingHostAllocator (via pin_memory=True)
- Djinn manages only the POLICY (when to swap), not the MECHANISM (how to allocate)
- Track session_id -> cpu_tensor mapping for lifecycle management
- Thread-safe with lock protection

Key Insight:
- PyTorch's allocator handles fragmentation via slab allocation
- We don't need to build our own bump-pointer allocator
- Focus on ADMISSION CONTROL (prefill) to prevent OOM, not memory management
"""

import logging
import threading
import torch
from typing import Dict, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class SwapMapping:
    """Track a swapped KV cache."""
    session_id: str
    cpu_tensor: torch.Tensor  # The pinned CPU tensor
    size_bytes: int           # Size of KV cache in bytes
    gpu_device: int           # GPU device ID this was swapped from
    timestamp: float          # When swapped


class HostSwapPool:
    """
    Manages pinned host memory for swapped KV caches using PyTorch's allocator.
    
    This is a POLICY layer (when to swap) not a MECHANISM layer (how to allocate).
    PyTorch's CachingHostAllocator handles the actual memory management.
    """
    
    def __init__(self, pool_size_gb: float = 32.0):
        """
        Initialize host swap pool (manages policy, not allocation).
        
        Args:
            pool_size_gb: Target pool size (informational, PyTorch manages actual allocation)
        """
        self.pool_size_gb = pool_size_gb
        self.pool_size_bytes = int(pool_size_gb * 1024**3)
        self._lock = threading.Lock()
        
        # Track swapped sessions
        self.swapped_sessions: Dict[str, SwapMapping] = {}
        
        # Statistics
        self.stats = {
            "swaps_performed": 0,
            "restores_performed": 0,
            "total_swapped_bytes": 0,
            "max_concurrent_swapped_mb": 0.0,
            "current_swapped_mb": 0.0,
            "swap_errors": 0,
            "restore_errors": 0,
        }
        
        logger.info(
            f"✅ HostSwapPool v2 initialized (SIMPLIFIED): "
            f"Using PyTorch's CachingHostAllocator for actual allocation, "
            f"Djinn manages policy (target: {pool_size_gb:.1f}GB)"
        )
    
    def allocate_and_swap(self, session_id: str, kv_cache_gpu, 
                         gpu_device: int = 0) -> int:
        """
        Swap KV cache from GPU to pinned host memory.
        
        Supports both raw Tensors and HuggingFace DynamicCache objects.
        
        SYNCHRONIZATION: This synchronizes the GPU stream to ensure data is fully copied
        before returning. This is safe because:
        1. Swap happens during idle "Act" phase (not on critical path)
        2. Correctness > micro-optimization for an OS
        3. PCIe transfer dominates (synchronize cost is negligible)
        
        Args:
            session_id: Session identifier
            kv_cache_gpu: KV cache (Tensor or DynamicCache) on GPU
            gpu_device: GPU device this is being swapped from
            
        Returns:
            Bytes swapped
            
        Raises:
            RuntimeError: If swap fails
        """
        with self._lock:
            if session_id in self.swapped_sessions:
                raise RuntimeError(f"Session {session_id} already swapped")
            
            try:
                # Handle DynamicCache: flatten to single tensor
                from transformers.modeling_outputs import Cache
                if isinstance(kv_cache_gpu, Cache):
                    # Flatten DynamicCache to a single tensor
                    flat_kv = self._flatten_dynamic_cache(kv_cache_gpu)
                else:
                    flat_kv = kv_cache_gpu
                
                # Allocate pinned memory (PyTorch's allocator handles it)
                cpu_buffer = torch.empty_like(flat_kv, pin_memory=True)
                
                # Copy KV from GPU to pinned CPU memory
                cpu_buffer.copy_(flat_kv, non_blocking=True)
                
                # ✅ SYNCHRONIZE: Ensure copy is complete before returning
                # This prevents race conditions if caller deletes GPU tensor
                torch.cuda.current_stream().synchronize()
                
                # Track the swap
                size_bytes = kv_cache_gpu.numel() * kv_cache_gpu.element_size()
                mapping = SwapMapping(
                    session_id=session_id,
                    cpu_tensor=cpu_buffer,
                    size_bytes=size_bytes,
                    gpu_device=gpu_device,
                    timestamp=time.time(),
                )
                self.swapped_sessions[session_id] = mapping
                
                # Update stats
                self.stats["swaps_performed"] += 1
                self.stats["total_swapped_bytes"] += size_bytes
                self.stats["current_swapped_mb"] = sum(
                    m.size_bytes for m in self.swapped_sessions.values()
                ) / (1024**2)
                self.stats["max_concurrent_swapped_mb"] = max(
                    self.stats["max_concurrent_swapped_mb"],
                    self.stats["current_swapped_mb"]
                )
                
                logger.info(
                    f"✅ Evicted KV to host: session_id={session_id[:12]}, "
                    f"size={size_bytes/1024**2:.1f}MB, "
                    f"total_swapped={self.stats['current_swapped_mb']:.1f}MB"
                )
                
                return size_bytes
                
            except Exception as e:
                self.stats["swap_errors"] += 1
                logger.error(f"Failed to swap session {session_id}: {e}")
                raise
    
    def _flatten_dynamic_cache(self, dyn_cache):
        """Flatten a HuggingFace DynamicCache to a single tensor."""
        try:
            # DynamicCache has key/value layer tuples
            # Flatten all of them into one tensor
            flat_list = []
            if hasattr(dyn_cache, 'key_cache'):
                for layer_kv in dyn_cache.key_cache:
                    if layer_kv is not None:
                        flat_list.append(layer_kv.flatten())
            if hasattr(dyn_cache, 'value_cache'):
                for layer_kv in dyn_cache.value_cache:
                    if layer_kv is not None:
                        flat_list.append(layer_kv.flatten())
            
            if not flat_list:
                raise RuntimeError("DynamicCache is empty")
            
            # Concatenate all flattened tensors
            return torch.cat(flat_list)
        except Exception as e:
            logger.error(f"Failed to flatten DynamicCache: {e}")
            raise
    
    def restore_and_copy(self, session_id: str, gpu_device: int = 0) -> int:
        """
        Restore KV cache from pinned host memory back to GPU.
        
        Args:
            session_id: Session to restore
            gpu_device: GPU device to restore to
            
        Returns:
            Bytes restored, or 0 if not found/not swapped
            
        Raises:
            RuntimeError: If restore fails
        """
        with self._lock:
            mapping = self.swapped_sessions.get(session_id)
            if mapping is None:
                logger.debug(f"Session {session_id} not swapped, skipping restore")
                return 0
            
            try:
                # Allocate new GPU tensor
                gpu_tensor = torch.empty_like(mapping.cpu_tensor, device=f'cuda:{gpu_device}')
                
                # Copy from pinned CPU back to GPU
                gpu_tensor.copy_(mapping.cpu_tensor, non_blocking=True)
                
                # Synchronize
                torch.cuda.current_stream().synchronize()
                
                # Free the pinned memory by deleting the CPU tensor
                # PyTorch's allocator will reclaim this memory for future allocations
                size_bytes = mapping.size_bytes
                del self.swapped_sessions[session_id]
                
                # Update stats
                self.stats["restores_performed"] += 1
                self.stats["current_swapped_mb"] = sum(
                    m.size_bytes for m in self.swapped_sessions.values()
                ) / (1024**2)
                
                logger.info(
                    f"✅ Restored KV from host: session_id={session_id[:12]}, "
                    f"size={size_bytes/1024**2:.1f}MB, "
                    f"total_swapped={self.stats['current_swapped_mb']:.1f}MB"
                )
                
                # Return the new GPU tensor for the caller to use
                return size_bytes
                
            except Exception as e:
                self.stats["restore_errors"] += 1
                logger.error(f"Failed to restore session {session_id}: {e}")
                raise
    
    def get_swapped_tensor(self, session_id: str) -> torch.Tensor:
        """Get the pinned CPU tensor for a swapped session (for inspection/testing)."""
        with self._lock:
            mapping = self.swapped_sessions.get(session_id)
            if mapping:
                return mapping.cpu_tensor
            return None
    
    def is_swapped(self, session_id: str) -> bool:
        """Check if session is currently swapped."""
        with self._lock:
            return session_id in self.swapped_sessions
    
    def get_stats(self) -> Dict:
        """Get statistics about swap pool usage."""
        with self._lock:
            return dict(self.stats)
    
    def clear(self) -> None:
        """Clear all swapped sessions (for testing/reset)."""
        with self._lock:
            self.swapped_sessions.clear()
            self.stats["current_swapped_mb"] = 0.0
            logger.info("Cleared host swap pool")


# Global singleton instance
_global_swap_pool: Optional[HostSwapPool] = None


def get_swap_pool(pool_size_gb: float = 32.0) -> HostSwapPool:
    """Get or create global swap pool singleton."""
    global _global_swap_pool
    
    if _global_swap_pool is None:
        _global_swap_pool = HostSwapPool(pool_size_gb=pool_size_gb)
    
    return _global_swap_pool

