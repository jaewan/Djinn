"""
Host Swap Pool: Pinned host memory management for KV cache eviction.

Purpose:
- Provide large pinned memory pool on CPU for swapped KV caches
- Manage allocation/deallocation without fragmentation
- Track which sessions are swapped to host
- Enable fast restore with async DMA transfers

Architecture:
- Pre-allocate configurable pinned memory pool at startup
- Use bump-pointer allocator for deterministic allocation
- Track session_id -> (host_offset, size_bytes) mapping
- Thread-safe with lock protection

Key Performance:
- Pinned memory: enables 24GB/s DMA (vs 4GB/s pageable)
- PCIe Gen4: 8GB KV cache takes ~333ms to transfer
- Fits within agent "Act" phase duration (10+ seconds typical)

Integration:
- Initialized during server startup
- Callbacks from SemanticActivityTracker trigger swap/restore
- KVSessionManager queries swap state
"""

import logging
import threading
import torch
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SwapMapping:
    """Track a swapped KV cache."""
    session_id: str
    host_offset: int      # Offset in pinned pool
    size_bytes: int       # Size of KV cache
    gpu_device: int       # GPU device ID this was swapped from
    timestamp: float      # When swapped (for LRU if needed)


class HostSwapPool:
    """
    Manages pinned host memory for swapped KV caches.
    
    Provides:
    - Pre-allocated pinned memory buffer
    - Session-based allocation tracking
    - Bump-pointer allocation strategy
    - Thread-safe access
    """
    
    def __init__(self, pool_size_gb: float = 32.0, alignment: int = 256):
        """
        Initialize host swap pool.
        
        Args:
            pool_size_gb: Size of pinned memory pool in GB (default: 32GB)
            alignment: Byte alignment for allocations (default: 256)
        """
        self.pool_size_bytes = int(pool_size_gb * 1024**3)
        self.alignment = alignment
        self._lock = threading.Lock()
        
        # Allocate pinned memory pool
        try:
            pool_tensor = torch.empty(self.pool_size_bytes, dtype=torch.uint8)
            self.pool = pool_tensor.pin_memory()
            logger.info(f"✅ Allocated pinned host swap pool: {pool_size_gb:.1f}GB")
        except RuntimeError as e:
            logger.error(f"❌ Failed to allocate pinned memory: {e}")
            raise RuntimeError(
                f"Cannot allocate {pool_size_gb}GB pinned memory. "
                f"Check ulimit -l and system available RAM."
            ) from e
        
        # Allocation tracking
        self.current_offset = 0
        self.swapped_sessions: Dict[str, SwapMapping] = {}
        
        # For fragmentation mitigation: track freed regions for potential compaction
        self._freed_regions: list[Tuple[int, int]] = []  # (offset, size) tuples
        
        # Statistics
        self.stats = {
            "swaps_performed": 0,
            "restores_performed": 0,
            "total_swapped_bytes": 0,
            "max_concurrent_swapped_mb": 0.0,
            "swap_errors": 0,
            "restore_errors": 0,
            "fragmentation_events": 0,
        }
        
        logger.info(
            f"HostSwapPool initialized: "
            f"pool_size={pool_size_gb:.1f}GB, alignment={alignment}B"
        )
    
    def allocate(self, session_id: str, size_bytes: int, gpu_device: int = 0) -> Tuple[int, torch.Tensor]:
        """
        Allocate space in swap pool for a session's KV cache.
        
        Args:
            session_id: Session identifier
            size_bytes: Size of KV cache to swap
            gpu_device: GPU device this is being swapped from
            
        Returns:
            (host_offset, tensor_view)
            
        Raises:
            RuntimeError: If insufficient space or session already swapped
        """
        with self._lock:
            if session_id in self.swapped_sessions:
                raise RuntimeError(f"Session {session_id} already swapped to host")
            
            # ✅ DEFRAGMENTATION: Try to reuse freed regions before bump pointer
            allocated_offset = None
            
            for idx, (free_offset, free_size) in enumerate(self._freed_regions):
                aligned_free_offset = (free_offset + self.alignment - 1) & ~(self.alignment - 1)
                alignment_overhead = aligned_free_offset - free_offset
                available_after_align = free_size - alignment_overhead
                
                if available_after_align >= size_bytes:
                    allocated_offset = aligned_free_offset
                    
                    # Update the freed region
                    remainder_offset = aligned_free_offset + size_bytes
                    remainder_size = free_size - alignment_overhead - size_bytes
                    
                    if remainder_size > 0:
                        self._freed_regions[idx] = (remainder_offset, remainder_size)
                    else:
                        self._freed_regions.pop(idx)
                    
                    logger.debug(f"Reusing freed region: offset={allocated_offset}, size={size_bytes} bytes")
                    break
            
            # If no freed region fits, use bump pointer
            if allocated_offset is None:
                aligned_offset = (self.current_offset + self.alignment - 1) & ~(self.alignment - 1)
            
            # Check capacity
            if aligned_offset + size_bytes > self.pool_size_bytes:
                available = self.pool_size_bytes - aligned_offset
                raise RuntimeError(
                    f"Swap pool exhausted: requested {size_bytes} bytes, "
                    f"available {available} bytes in {self.pool_size_bytes/1024**3:.1f}GB pool"
                )
                
                allocated_offset = aligned_offset
                self.current_offset = aligned_offset + size_bytes
            
            # Create view and track allocation
            view = self.pool[allocated_offset:allocated_offset + size_bytes]
            
            import time
            mapping = SwapMapping(
                session_id=session_id,
                host_offset=allocated_offset,
                size_bytes=size_bytes,
                gpu_device=gpu_device,
                timestamp=time.time(),
            )
            self.swapped_sessions[session_id] = mapping
            
            # Update stats
            self.stats["swaps_performed"] += 1
            self.stats["total_swapped_bytes"] += size_bytes
            current_swapped_mb = sum(m.size_bytes for m in self.swapped_sessions.values()) / 1024**2
            self.stats["max_concurrent_swapped_mb"] = max(
                self.stats["max_concurrent_swapped_mb"],
                current_swapped_mb
            )
            
            logger.debug(
                f"Allocated swap space: session_id={session_id}, "
                f"offset={allocated_offset}, size={size_bytes/1024**2:.1f}MB"
            )
            
            return allocated_offset, view
    
    def get_mapping(self, session_id: str) -> Optional[SwapMapping]:
        """
        Get swap mapping for a session (if swapped).
        
        Args:
            session_id: Session to query
            
        Returns:
            SwapMapping if swapped, None otherwise
        """
        with self._lock:
            return self.swapped_sessions.get(session_id)
    
    def is_swapped(self, session_id: str) -> bool:
        """
        Check if session is currently swapped to host.
        
        Args:
            session_id: Session to check
            
        Returns:
            True if swapped, False otherwise
        """
        with self._lock:
            return session_id in self.swapped_sessions
    
    def free(self, session_id: str) -> int:
        """
        Free swap allocation for a session.
        
        Tracks freed regions to mitigate fragmentation and enable compaction.
        
        Args:
            session_id: Session to free
            
        Returns:
            Bytes freed
        """
        with self._lock:
            mapping = self.swapped_sessions.pop(session_id, None)
            if mapping is None:
                return 0
            
            # Track freed region for potential compaction
            self._freed_regions.append((mapping.host_offset, mapping.size_bytes))
            
            # Check if we can compact the pool (if freed region is at the end)
            if mapping.host_offset + mapping.size_bytes == self.current_offset:
                # This freed region is at the end - we can reclaim it immediately
                self.current_offset = mapping.host_offset
                self._freed_regions.remove((mapping.host_offset, mapping.size_bytes))
                logger.debug(f"Immediate compaction: freed end region {mapping.size_bytes} bytes")
            else:
                # Track fragmentation event for monitoring
                if len(self._freed_regions) > 1:
                    self.stats["fragmentation_events"] += 1
            
            self.stats["restores_performed"] += 1
            
            logger.debug(
                f"Freed swap space: session_id={session_id}, "
                f"size={mapping.size_bytes/1024**2:.1f}MB, "
                f"fragmented_regions={len(self._freed_regions)}"
            )
            
            return mapping.size_bytes
    
    def get_host_view(self, session_id: str) -> Optional[torch.Tensor]:
        """
        Get tensor view into swap pool for a session.
        
        Used by KVSessionManager to read/write swapped data.
        
        Args:
            session_id: Session to access
            
        Returns:
            Tensor view if swapped, None otherwise
        """
        with self._lock:
            mapping = self.swapped_sessions.get(session_id)
            if mapping is None:
                return None
            
            view = self.pool[mapping.host_offset:mapping.host_offset + mapping.size_bytes]
            return view
    
    def get_stats(self) -> Dict:
        """
        Get pool statistics.
        
        Returns:
            Dictionary of stats
        """
        with self._lock:
            current_used = self.current_offset
            current_swapped = sum(m.size_bytes for m in self.swapped_sessions.values())
            
            return {
                **self.stats,
                "pool_size_bytes": self.pool_size_bytes,
                "pool_size_gb": self.pool_size_bytes / 1024**3,
                "current_allocated_bytes": current_used,
                "current_allocated_mb": current_used / 1024**2,
                "current_swapped_bytes": current_swapped,
                "current_swapped_mb": current_swapped / 1024**2,
                "active_swapped_sessions": len(self.swapped_sessions),
                "utilization_percent": (current_used / self.pool_size_bytes * 100)
                    if self.pool_size_bytes > 0 else 0,
            }
    
    def clear(self) -> None:
        """
        Clear all swaps and reset pool allocator.
        
        Used during server shutdown or emergency cleanup.
        Marks all sessions as no longer swapped.
        """
        with self._lock:
            cleared_count = len(self.swapped_sessions)
            self.swapped_sessions.clear()
            self.current_offset = 0
            logger.info(f"Cleared host swap pool ({cleared_count} sessions)")


# Global singleton instance
_global_swap_pool: Optional[HostSwapPool] = None


def get_swap_pool(pool_size_gb: float = 32.0) -> HostSwapPool:
    """Get or create global swap pool singleton."""
    global _global_swap_pool
    
    if _global_swap_pool is None:
        _global_swap_pool = HostSwapPool(pool_size_gb=pool_size_gb)
    
    return _global_swap_pool

