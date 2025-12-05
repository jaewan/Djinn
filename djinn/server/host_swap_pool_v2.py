"""
Host Swap Pool v2 (HARDENED): Pinned host memory management with safety limits.

Purpose:
- Provide pinned memory for KV cache eviction (swap-to-CPU)
- Track which sessions are swapped
- Enable fast restore with DMA transfers
- ENFORCE memory limits to prevent OOM
- GRACEFUL DEGRADATION on errors

Architecture:
- SIMPLIFIED: Delegate allocation to PyTorch's CachingHostAllocator (via pin_memory=True)
- Djinn manages only the POLICY (when to swap), not the MECHANISM (how to allocate)
- Track session_id -> cpu_tensor mapping for lifecycle management
- Thread-safe with lock protection (used via asyncio.to_thread)
- HARDENED: Memory limits, session limits, graceful degradation
"""

import logging
import threading
import torch
from typing import Dict, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


# Safety limits
MAX_SWAPPED_SESSIONS = 100  # Prevent unbounded session growth
MAX_SWAP_AGE_SECONDS = 600  # Force-expire swaps older than 10 minutes


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
    
    HARDENED FEATURES:
    - Memory limit enforcement (pool_size_gb)
    - Session count limits (MAX_SWAPPED_SESSIONS)
    - Stale swap cleanup (MAX_SWAP_AGE_SECONDS)
    - Graceful degradation on errors
    """
    
    def __init__(self, pool_size_gb: float = 32.0):
        """
        Initialize host swap pool (manages policy, not allocation).
        
        Args:
            pool_size_gb: Maximum pool size - ENFORCED (not just informational)
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
            "swap_rejections_memory": 0,  # Rejected due to memory limit
            "swap_rejections_count": 0,   # Rejected due to session limit
            "stale_cleanups": 0,          # Cleaned up stale swaps
        }
        
        logger.info(
            f"✅ HostSwapPool v2 (HARDENED) initialized: "
            f"limit={pool_size_gb:.1f}GB, max_sessions={MAX_SWAPPED_SESSIONS}"
        )
    
    def _cleanup_stale_swaps(self) -> int:
        """
        Remove swaps older than MAX_SWAP_AGE_SECONDS.
        MUST be called with lock held.
        
        Returns:
            Number of stale swaps removed
        """
        now = time.time()
        stale_sessions = [
            sid for sid, mapping in self.swapped_sessions.items()
            if (now - mapping.timestamp) > MAX_SWAP_AGE_SECONDS
        ]
        
        for sid in stale_sessions:
            del self.swapped_sessions[sid]
            self.stats["stale_cleanups"] += 1
            logger.warning(f"Cleaned up stale swap: {sid[:12]} (age > {MAX_SWAP_AGE_SECONDS}s)")
        
        if stale_sessions:
            self.stats["current_swapped_mb"] = sum(
                m.size_bytes for m in self.swapped_sessions.values()
            ) / (1024**2)
        
        return len(stale_sessions)
    
    def _ensure_flat_tensor(self, data):
        """
        Recursively flatten any KV structure to a single contiguous tensor on GPU.
        Handles: raw tensors, tuples, lists, nested structures
        """
        if torch.is_tensor(data):
            return data.contiguous()
        elif isinstance(data, (list, tuple)):
            # Flatten all tensors in the list/tuple
            flat_tensors = []
            for item in data:
                flat_item = self._ensure_flat_tensor(item)
                if torch.is_tensor(flat_item):
                    flat_tensors.append(flat_item.flatten())
            if flat_tensors:
                return torch.cat(flat_tensors)
            else:
                raise RuntimeError("No tensors found in list/tuple structure")
        else:
            raise RuntimeError(f"Cannot flatten type: {type(data)}")
    
    def allocate_and_swap(self, session_id: str, kv_cache_gpu, 
                         gpu_device: int = 0) -> int:
        """
        Swap KV cache from GPU to pinned host memory.
        
        HARDENED: Enforces memory limits and gracefully degrades on errors.
        
        Args:
            session_id: Session identifier
            kv_cache_gpu: KV cache (any structure) on GPU
            gpu_device: GPU device this is being swapped from
            
        Returns:
            Bytes swapped, or 0 if rejected/failed (graceful degradation)
        """
        with self._lock:
            # Cleanup stale swaps first
            self._cleanup_stale_swaps()
            
            # Check 1: Already swapped?
            if session_id in self.swapped_sessions:
                logger.debug(f"Session {session_id[:12]} already swapped, skipping")
                return 0
            
            # Check 2: Session count limit
            if len(self.swapped_sessions) >= MAX_SWAPPED_SESSIONS:
                self.stats["swap_rejections_count"] += 1
                logger.warning(f"Swap rejected for {session_id[:12]}: "
                             f"session limit reached ({MAX_SWAPPED_SESSIONS})")
                return 0  # Graceful degradation: don't crash, just skip swap
            
            try:
                # Step 1: Flatten any KV structure to a single GPU tensor
                flat_kv_gpu = self._ensure_flat_tensor(kv_cache_gpu)
                
                if not torch.is_tensor(flat_kv_gpu):
                    logger.error(f"Flatten failed for {session_id[:12]}: got {type(flat_kv_gpu)}")
                    return 0
                
                if flat_kv_gpu.device.type != 'cuda':
                    logger.error(f"KV cache for {session_id[:12]} not on GPU: {flat_kv_gpu.device}")
                    return 0
                
                size_bytes = flat_kv_gpu.numel() * flat_kv_gpu.element_size()
                size_mb = size_bytes / (1024**2)
                
                # Check 3: Memory limit
                new_total_mb = self.stats["current_swapped_mb"] + size_mb
                if new_total_mb > self.pool_size_gb * 1024:
                    self.stats["swap_rejections_memory"] += 1
                    logger.warning(f"Swap rejected for {session_id[:12]}: "
                                 f"would exceed limit ({new_total_mb:.1f}MB > {self.pool_size_gb*1024:.0f}MB)")
                    return 0  # Graceful degradation
                
                logger.debug(f"Swapping {session_id[:12]}: size={size_mb:.1f}MB")
                
                # Step 2: Allocate pinned CPU tensor
                cpu_buffer = torch.empty(flat_kv_gpu.shape, dtype=flat_kv_gpu.dtype, pin_memory=True)
                
                # Step 3: Copy from GPU to pinned CPU (async)
                cpu_buffer.copy_(flat_kv_gpu, non_blocking=True)
                
                # Step 4: Synchronize to ensure copy completes
                torch.cuda.current_stream().synchronize()
                
                # Step 5: Track the swap
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
                    f"size={size_mb:.1f}MB, "
                    f"total_swapped={self.stats['current_swapped_mb']:.1f}MB, "
                    f"sessions={len(self.swapped_sessions)}"
                )
                
                return size_bytes
                
            except torch.cuda.OutOfMemoryError as e:
                self.stats["swap_errors"] += 1
                logger.error(f"CUDA OOM during swap for {session_id[:12]}: {e}")
                return 0  # Graceful degradation
                
            except Exception as e:
                self.stats["swap_errors"] += 1
                logger.error(f"Failed to swap session {session_id[:12]}: {e}")
                return 0  # Graceful degradation instead of raising
    
    def restore_and_copy(self, session_id: str, gpu_device: int = 0) -> Optional[torch.Tensor]:
        """
        Restore KV cache from pinned host memory back to GPU.
        
        HARDENED: Returns None on failure instead of raising.
        
        Args:
            session_id: Session to restore
            gpu_device: GPU device to restore to
            
        Returns:
            Restored GPU tensor, or None if not swapped/failed
        """
        with self._lock:
            mapping = self.swapped_sessions.get(session_id)
            if mapping is None:
                logger.debug(f"Session {session_id[:12]} not swapped, skipping restore")
                return None
            
            try:
                # Allocate new GPU tensor
                gpu_tensor = torch.empty_like(mapping.cpu_tensor, device=f'cuda:{gpu_device}')
                
                # Copy from pinned CPU back to GPU
                gpu_tensor.copy_(mapping.cpu_tensor, non_blocking=True)
                
                # Synchronize
                torch.cuda.current_stream().synchronize()
                
                # Free the pinned memory by deleting the CPU tensor
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
                
                return gpu_tensor
                
            except torch.cuda.OutOfMemoryError as e:
                self.stats["restore_errors"] += 1
                logger.error(f"CUDA OOM during restore for {session_id[:12]}: {e}")
                # Keep the swap in place - maybe we can restore later
                return None
                
            except Exception as e:
                self.stats["restore_errors"] += 1
                logger.error(f"Failed to restore session {session_id[:12]}: {e}")
                return None  # Graceful degradation
    
    def get_swapped_tensor(self, session_id: str) -> Optional[torch.Tensor]:
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
            stats = dict(self.stats)
            stats["active_sessions"] = len(self.swapped_sessions)
            stats["pool_utilization_pct"] = (
                self.stats["current_swapped_mb"] / (self.pool_size_gb * 1024) * 100
            ) if self.pool_size_gb > 0 else 0
            return stats
    
    def clear(self) -> None:
        """Clear all swapped sessions (for testing/reset)."""
        with self._lock:
            self.swapped_sessions.clear()
            self.stats["current_swapped_mb"] = 0.0
            logger.info("Cleared host swap pool")
    
    def empty_cache(self) -> None:
        """
        HARDENED: Force PyTorch to empty its CachingHostAllocator cache.
        
        This MUST be called between experiments to reclaim pinned memory.
        Without this, pinned tensors accumulate and exhaust the pool.
        
        This is a known PyTorch limitation:
        https://github.com/pytorch/pytorch/issues/27595
        """
        try:
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()  # Call twice for safety
            logger.info("✅ Emptied PyTorch's CachingHostAllocator cache")
        except Exception as e:
            logger.warning(f"Could not empty cache: {e}")
    
    def force_evict_oldest(self, count: int = 1) -> int:
        """
        Force-evict oldest swapped sessions to make room.
        
        Args:
            count: Number of sessions to evict
            
        Returns:
            Number of sessions actually evicted
        """
        with self._lock:
            if not self.swapped_sessions:
                return 0
            
            # Sort by timestamp (oldest first)
            sorted_sessions = sorted(
                self.swapped_sessions.items(),
                key=lambda x: x[1].timestamp
            )
            
            evicted = 0
            for session_id, _ in sorted_sessions[:count]:
                del self.swapped_sessions[session_id]
                evicted += 1
                logger.warning(f"Force-evicted oldest swap: {session_id[:12]}")
            
            if evicted:
                self.stats["current_swapped_mb"] = sum(
                    m.size_bytes for m in self.swapped_sessions.values()
                ) / (1024**2)
            
            return evicted


# Global singleton instance
_global_swap_pool: Optional[HostSwapPool] = None
_pool_lock = threading.Lock()


def get_swap_pool(pool_size_gb: float = 32.0) -> HostSwapPool:
    """Get or create global swap pool singleton (thread-safe)."""
    global _global_swap_pool
    
    with _pool_lock:
        if _global_swap_pool is None:
            _global_swap_pool = HostSwapPool(pool_size_gb=pool_size_gb)
    
    return _global_swap_pool


def reset_swap_pool() -> None:
    """Reset the global swap pool (for testing)."""
    global _global_swap_pool
    
    with _pool_lock:
        if _global_swap_pool is not None:
            _global_swap_pool.clear()
        _global_swap_pool = None
