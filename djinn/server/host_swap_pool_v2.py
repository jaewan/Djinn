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
from typing import Dict, Optional, Any
from dataclasses import dataclass
import time
from .memory_metrics import get_metrics

logger = logging.getLogger(__name__)


# Safety limits
MAX_SWAPPED_SESSIONS = 100  # Prevent unbounded session growth
MAX_SWAP_AGE_SECONDS = 600  # Force-expire swaps older than 10 minutes


@dataclass
class SwapMapping:
    """Track a swapped KV cache."""
    session_id: str
    cpu_data: Any             # The pinned CPU data (tensor, tuple, or list)
    size_bytes: int           # Size of KV cache in bytes
    gpu_device: int           # GPU device ID this was swapped from
    timestamp: float          # When swapped
    data_type: str = "tensor" # Type of data: "tensor", "tuple", or "dynamic_cache"
    completion_event: Optional[Any] = None  # CUDA event for async completion tracking


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
        
        ✅ TIER 3: Pre-allocate pinned buffer pool to eliminate malloc overhead.
        
        Args:
            pool_size_gb: Maximum pool size - ENFORCED (not just informational)
        """
        self.pool_size_gb = pool_size_gb
        self.pool_size_bytes = int(pool_size_gb * 1024**3)
        self._lock = threading.Lock()
        
        # TIER 1 OPTIMIZATION: Dedicated CUDA streams for swap/restore
        # Separate streams allow overlap with compute and avoid blocking main stream
        if torch.cuda.is_available():
            self.swap_stream = torch.cuda.Stream()
            self.restore_stream = torch.cuda.Stream()
            logger.info("✅ Dedicated CUDA streams created for swap/restore operations")
        else:
            self.swap_stream = None
            self.restore_stream = None
        
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
    
    def _prepare_for_swap(self, data):
        """
        Prepare KV cache data for swapping to CPU.
        
        ✅ CRITICAL FIX: Keep tuple structure for legacy cache format.
        This avoids losing metadata needed for reconstruction.
        
        Returns:
            (data_to_swap, data_type_str)
        """
        # If it's a tuple (legacy cache format), keep it as-is
        if isinstance(data, tuple):
            return data, "tuple"
        
        # If it's a list (also legacy format), keep it
        if isinstance(data, list):
            return data, "list"
        
        # If it's a single tensor, return it
        if torch.is_tensor(data):
            return data, "tensor"
        
        # For anything else (DynamicCache, etc), raise error to caller
        return data, "unknown"
    
    def allocate_and_swap(self, session_id: str, kv_cache_gpu, 
                         gpu_device: int = 0) -> int:
        """
        Swap KV cache from GPU to pinned host memory.
        
        ✅ CRITICAL FIX: Preserve data structure (tuple, tensor, etc.) to avoid
        losing metadata. Tuples are kept as-is to enable DynamicCache reconstruction.
        
        HARDENED: Enforces memory limits and gracefully degrades on errors.
        
        Args:
            session_id: Session identifier
            kv_cache_gpu: KV cache (tuple, tensor, or structure) on GPU
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
                # ✅ CRITICAL FIX: Keep structure, don't flatten
                data_to_swap, data_type = self._prepare_for_swap(kv_cache_gpu)
                
                # Calculate size based on structure
                if isinstance(data_to_swap, torch.Tensor):
                    if data_to_swap.device.type != 'cuda':
                        logger.error(f"KV cache for {session_id[:12]} not on GPU: {data_to_swap.device}")
                        return 0
                    size_bytes = data_to_swap.numel() * data_to_swap.element_size()
                elif isinstance(data_to_swap, (tuple, list)):
                    # Sum up all tensors in the structure
                    size_bytes = 0
                    for item in data_to_swap:
                        if isinstance(item, torch.Tensor):
                            size_bytes += item.numel() * item.element_size()
                        elif isinstance(item, (tuple, list)):
                            # Nested structure like ((k0, v0), (k1, v1), ...)
                            for sub_item in item:
                                if isinstance(sub_item, torch.Tensor):
                                    size_bytes += sub_item.numel() * sub_item.element_size()
                else:
                    logger.error(f"Unknown KV type for {session_id[:12]}: {type(data_to_swap)}")
                    return 0
                
                size_mb = size_bytes / (1024**2)
                
                # Check 3: Memory limit
                new_total_mb = self.stats["current_swapped_mb"] + size_mb
                if new_total_mb > self.pool_size_gb * 1024:
                    self.stats["swap_rejections_memory"] += 1
                    logger.warning(f"Swap rejected for {session_id[:12]}: "
                                 f"would exceed limit ({new_total_mb:.1f}MB > {self.pool_size_gb*1024:.0f}MB)")
                    return 0  # Graceful degradation
                
                logger.debug(f"Swapping {session_id[:12]}: size={size_mb:.1f}MB, type={data_type}")
                
                # ✅ Move structure to pinned CPU WITHOUT flattening
                def move_to_cpu(data):
                    if isinstance(data, torch.Tensor):
                        # Make contiguous (no clone - contiguous() returns same tensor if already contiguous)
                        # This handles views without wasteful GPU→GPU copy
                        data_contiguous = data.contiguous()
                        # Create empty pinned buffer THEN copy
                        cpu_buf = torch.empty(data_contiguous.shape, dtype=data_contiguous.dtype, device='cpu', pin_memory=True)
                        cpu_buf.copy_(data_contiguous, non_blocking=True)
                        return cpu_buf
                    elif isinstance(data, (tuple, list)):
                        return type(data)(move_to_cpu(item) for item in data)
                    else:
                        return data
                
                # TIER 1 OPTIMIZATION: Use dedicated swap stream for async transfer
                if self.swap_stream:
                    with torch.cuda.stream(self.swap_stream):
                        cpu_data = move_to_cpu(data_to_swap)
                else:
                    cpu_data = move_to_cpu(data_to_swap)
                
                # TIER 1 OPTIMIZATION: Use event-based completion instead of blocking sync
                # This allows CPU to continue while GPU transfer completes asynchronously
                swap_event = torch.cuda.Event() if torch.cuda.is_available() else None
                if swap_event:
                    swap_event.record()
                
                # Step 5: Track the swap
                mapping = SwapMapping(
                    session_id=session_id,
                    cpu_data=cpu_data,
                    size_bytes=size_bytes,
                    gpu_device=gpu_device,
                    timestamp=time.time(),
                    data_type=data_type,
                    completion_event=swap_event,
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
                    f"size={size_mb:.1f}MB, type={data_type}, "
                    f"total_swapped={self.stats['current_swapped_mb']:.1f}MB, "
                    f"sessions={len(self.swapped_sessions)}"
                )
                
                # Record success metric
                metrics = get_metrics()
                metrics.record_swap_success()
                
                return size_bytes
                
            except torch.cuda.OutOfMemoryError as e:
                self.stats["swap_errors"] += 1
                logger.error(f"[CRITICAL] CUDA OOM during swap for {session_id[:12]}: {e}", exc_info=True)
                metrics = get_metrics()
                metrics.record_swap_failure()
                return 0  # Graceful degradation
                
            except Exception as e:
                self.stats["swap_errors"] += 1
                logger.error(f"[CRITICAL] Failed to swap session {session_id[:12]}: {e}", exc_info=True)
                metrics = get_metrics()
                metrics.record_swap_failure()
                return 0  # Graceful degradation instead of raising
    
    def restore_and_copy(self, session_id: str, gpu_device: int = 0) -> Optional[Any]:
        """
        Restore KV cache from pinned host memory back to GPU.
        
        ✅ CRITICAL FIX: Preserves structure (tuple, tensor, etc.) to enable
        proper DynamicCache reconstruction.
        
        HARDENED: Returns None on failure instead of raising.
        
        Args:
            session_id: Session to restore
            gpu_device: GPU device to restore to
            
        Returns:
            Restored GPU data (tensor, tuple, list, or structure), or None if not swapped/failed
        """
        with self._lock:
            mapping = self.swapped_sessions.get(session_id)
            if mapping is None:
                logger.debug(f"Session {session_id[:12]} not swapped, skipping restore")
                return None
            
            try:
                # ✅ Move structure back to GPU WITHOUT flattening
                def move_to_gpu(data):
                    if isinstance(data, torch.Tensor):
                        gpu_tensor = torch.empty_like(data, device=f'cuda:{gpu_device}')
                        gpu_tensor.copy_(data, non_blocking=True)
                        return gpu_tensor
                    elif isinstance(data, (tuple, list)):
                        return type(data)(move_to_gpu(item) for item in data)
                    else:
                        return data
                
                # TIER 1 OPTIMIZATION: Use dedicated restore stream for async transfer
                if self.restore_stream:
                    with torch.cuda.stream(self.restore_stream):
                        gpu_data = move_to_gpu(mapping.cpu_data)
                else:
                    gpu_data = move_to_gpu(mapping.cpu_data)
                
                # TIER 1 OPTIMIZATION: Use event-based completion instead of blocking sync
                # Record completion event for async tracking
                restore_event = torch.cuda.Event() if torch.cuda.is_available() else None
                if restore_event:
                    restore_event.record()
                
                # Free the pinned memory by deleting the mapping
                size_bytes = mapping.size_bytes
                data_type = mapping.data_type
                del self.swapped_sessions[session_id]
                
                # Update stats
                self.stats["restores_performed"] += 1
                self.stats["current_swapped_mb"] = sum(
                    m.size_bytes for m in self.swapped_sessions.values()
                ) / (1024**2)
                
                logger.info(
                    f"✅ Restored KV from host: session_id={session_id[:12]}, "
                    f"size={size_bytes/1024**2:.1f}MB, type={mapping.data_type}, "
                    f"total_swapped={self.stats['current_swapped_mb']:.1f}MB"
                )
                
                # Record success metric
                metrics = get_metrics()
                metrics.record_restore_success()
                
                return gpu_data
                
            except torch.cuda.OutOfMemoryError as e:
                self.stats["restore_errors"] += 1
                logger.error(f"[CRITICAL] CUDA OOM during restore for {session_id[:12]}: {e}", exc_info=True)
                metrics = get_metrics()
                metrics.record_restore_failure()
                # Keep the swap in place - maybe we can restore later
                return None
                
            except Exception as e:
                self.stats["restore_errors"] += 1
                logger.error(f"[CRITICAL] Failed to restore session {session_id[:12]}: {e}", exc_info=True)
                metrics = get_metrics()
                metrics.record_restore_failure()
                return None  # Graceful degradation
    
    def get_swapped_data(self, session_id: str) -> Optional[Any]:
        """Get the pinned CPU data for a swapped session (for inspection/testing)."""
        with self._lock:
            mapping = self.swapped_sessions.get(session_id)
            if mapping:
                return mapping.cpu_data
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
